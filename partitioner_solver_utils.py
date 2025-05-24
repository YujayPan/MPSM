# partitioner_solver_utils.py
import torch
import os
import re
import logging
import math
import numpy as np
from scipy.spatial.distance import cdist
from utils import get_model, num_param, VRP_DATA_FORMAT, get_env, load_dataset
from models import MOEModel, MOEModel_Light

logger = logging.getLogger(__name__)


# Define default parameters based on train.py defaults
DEFAULT_MODEL_PARAMS = {
    "model_type": "MOE", 
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128**(1/2),
    "encoder_layer_num": 6,
    "decoder_layer_num": 1,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10.0,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",
    "num_experts": 4,       # Default for MOE, adjust if needed based on type
    "norm": "instance",
    "norm_loc": "norm_last",
    "expert_loc": ['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'],
    "topk": 2,
    "routing_level": "node",
    "routing_method": "input_choice",
    "problem": "Train_ALL"  # Default problem type reflecting train.py
}


def load_moe_model(checkpoint_path, device, model_type: str | None = None, model_params: dict | None = None):
    """Loads a specific MoE model checkpoint, inferring type and using default params if needed.

    Args:
        checkpoint_path (str): Path to the pre-trained model checkpoint (.pt file).
        device (torch.device): The device to load the model onto (e.g., 'cuda', 'cpu').
        model_type (str | None, optional): The type of model (e.g., "MOE", "MOE_LIGHT").
                                           If None, attempts to infer from path. Defaults to None.
        model_params (dict | None, optional): Dictionary of model hyperparameters.
                                              If None, uses DEFAULT_MODEL_PARAMS. Defaults to None.

    Returns:
        torch.nn.Module or None: The loaded PyTorch model in evaluation mode, or None if loading fails.
    """
    logger.info(f"Attempting to load model from checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None
    if get_model is None: # Check if import failed
        logger.error("Error: 'get_model' function not available. Cannot load model.")
        return None

    # --- Determine Model Type ---
    inferred_type = None
    if model_type is None:
        path_lower = checkpoint_path.lower().replace('\\', '/') # Normalize path separators
        if 'mvmoe' in path_lower and '_light_' in path_lower:
            inferred_type = "MOE_LIGHT"
            logger.info("Inferred model type as MOE_LIGHT from path.")
        elif 'moe_' in path_lower and '_light_' in path_lower: # Check general pattern too
            inferred_type = "MOE_LIGHT"
            logger.info("Inferred model type as MOE_LIGHT from path.")
        elif 'mvmoe' in path_lower or 'moe_' in path_lower:
             inferred_type = "MOE"
             logger.info("Inferred model type as MOE from path.")
        elif 'mtl' in path_lower:
             inferred_type = "MTL"
             logger.info("Inferred model type as MTL from path.")
        elif 'single' in path_lower:
             inferred_type = "SINGLE"
             logger.info("Inferred model type as SINGLE from path.")
        else:
             # Fallback if path doesn't contain expected pattern
             inferred_type = "MOE" # Default assumption for fine-tuning partitioner
             logger.warning(f"Could not infer model type from path '{checkpoint_path}'. Assuming 'MOE'. Provide 'model_type' argument for clarity.")
        model_type_to_use = inferred_type
    else:
        model_type_to_use = model_type
        logger.info(f"Using provided model type: {model_type_to_use}")

    # --- Determine Model Params ---
    if model_params is None:
        model_params_to_use = DEFAULT_MODEL_PARAMS.copy()
        logger.info("Using default model parameters initially.")
        
        # --- Attempt to infer num_experts from path if not provided ---
        path_lower_for_experts = checkpoint_path.lower().replace('\\', '/')
        # Use regex to find patterns like _4e_ or _8e_
        match_experts = re.search(r'_(\d+)e_', path_lower_for_experts)
        if match_experts:
            try:
                inferred_num_experts = int(match_experts.group(1))
                model_params_to_use['num_experts'] = inferred_num_experts
                logger.info(f"Inferred num_experts={inferred_num_experts} from checkpoint path.")
            except (ValueError, IndexError):
                logger.warning(f"Found expert pattern '{match_experts.group(0)}' in path, but could not parse number. Using default num_experts: {model_params_to_use['num_experts']}.")
        else:
            logger.warning(f"Could not infer num_experts (_Xe_) from path '{checkpoint_path}'. Using default: {model_params_to_use['num_experts']}.")
        # --- End expert inference ---
        
        # Adjust default num_experts based on inferred type if necessary (though 4 is common for both)
        # if model_type_to_use == "MOE_LIGHT": model_params_to_use['num_experts'] = ... # Example: Keep this commented or adjust if needed
    else:
        model_params_to_use = model_params.copy()
        logger.info("Using provided model parameters.")


    try:
        # Explicitly set weights_only=False to load checkpoints saved with older PyTorch versions
        # or those containing non-tensor objects like numpy scalars.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Extract model state dict (handle different checkpoint structures)
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint: # common alternative key
            model_state_dict = checkpoint['state_dict']
        else:
            # Assume the checkpoint *is* the state dict if no common keys found
            model_state_dict = checkpoint
            logger.warning(f"Checkpoint keys 'model_state_dict' or 'state_dict' not found in {checkpoint_path}. Assuming the file is the state dict itself.")

        logger.info(f">> Checkpoint loaded successfully. Epoch: {checkpoint.get('epoch', 'N/A')}")

        # Update model params with problem from checkpoint if available
        final_model_params = model_params_to_use
        ckpt_args = checkpoint.get('args', None) # Check if args saved in ckpt
        if ckpt_args and 'problem' in ckpt_args:
            final_model_params['problem'] = ckpt_args['problem']
            logger.info(f">> Using problem type '{ckpt_args['problem']}' from checkpoint args for model instantiation.")
        elif 'problem' in checkpoint:
             # Fallback to top-level 'problem' key if args not present
             final_model_params['problem'] = checkpoint['problem']
             logger.info(f">> Using problem type '{checkpoint['problem']}' from checkpoint (top-level) for model instantiation.")
        else:
             logger.warning(f">> No 'problem' key found in checkpoint. Using '{final_model_params['problem']}' from parameters.")


        # Instantiate Model using determined type and params
        ModelClass = get_model(model_type_to_use)
        if ModelClass is None:
            logger.error(f"Error: get_model('{model_type_to_use}') returned None.")
            return None
        model = ModelClass(**final_model_params).to(device)
        # logger.info(f">> Model Class {type(model).__name__} instantiated.")

        # Load State Dictionary
        try:
            # Try strict loading first, as it's safer for models expected to match
            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=True)
            if missing_keys:
                logger.warning(f"Missing keys found during strict load: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys found during strict load: {unexpected_keys}")
            logger.info(">> Model state_dict loaded successfully (strict=True).")
        except RuntimeError as e:
            logger.warning(f"Strict loading failed ({e}). Attempting non-strict loading...")
            # Attempt loading with strict=False as a fallback
            try:
                missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
                logger.warning(">> Loaded state_dict with strict=False due to potential mismatch.")
                if missing_keys:
                    logger.warning(f"  Missing keys (strict=False): {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"  Unexpected keys (strict=False): {unexpected_keys}")
            except Exception as load_err:
                 logger.error(f"Failed to load state_dict even with strict=False: {load_err}")
                 return None

        # Set to evaluation mode
        model.eval()
        logger.info(">> Model set to evaluation mode.")
        # Optional: Print parameter count
        # if num_param: num_param(model)

        return model

    except FileNotFoundError: # Should be caught earlier, but keep for safety
        logger.error(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during model loading: {e}")
        return None


def create_subproblem_instance(original_instance_tuple, problem_type, subproblem_node_indices):
    """Creates a new VRP instance tuple for a given subproblem.

    Args:
        original_instance_tuple (tuple): The raw data tuple of the original VRP instance.
        problem_type (str): The type of the original VRP problem (e.g., "CVRP", "VRPTW").
        subproblem_node_indices (list[int]): A list of 1-based node indices belonging to this subproblem.

    Returns:
        tuple or None: A new VRP instance tuple containing only the depot and the specified
                       subproblem nodes, formatted according to problem_type. Returns None
                       if input is invalid or an error occurs.
    """
    if not original_instance_tuple or not subproblem_node_indices or problem_type not in VRP_DATA_FORMAT:
        logger.error(f"create_subproblem_instance: Invalid input - tuple: {original_instance_tuple is None}, indices: {subproblem_node_indices is None}, type: {problem_type}")
        return None

    format_keys = VRP_DATA_FORMAT[problem_type]
    min_expected_len = len(format_keys)

    if len(original_instance_tuple) < min_expected_len:
        logger.error(f"create_subproblem_instance: Original instance tuple length ({len(original_instance_tuple)}) is less than expected ({min_expected_len}) for {problem_type}.")
        return None

    try:
        # Create a mapping from key name to its index in the original tuple
        key_to_index = {key: idx for idx, key in enumerate(format_keys)}

        # Extract data from the original tuple using the mapping
        depot_xy = original_instance_tuple[key_to_index["depot_xy"]]
        original_node_xy = original_instance_tuple[key_to_index["node_xy"]]
        original_demand = original_instance_tuple[key_to_index["demand"]]
        capacity = original_instance_tuple[key_to_index["capacity"]]

        # Optional components
        route_limit = original_instance_tuple[key_to_index["route_limit"]] if "route_limit" in key_to_index else None
        original_service_time = original_instance_tuple[key_to_index["service_time"]] if "service_time" in key_to_index else None
        original_tw_start = original_instance_tuple[key_to_index["tw_start"]] if "tw_start" in key_to_index else None
        original_tw_end = original_instance_tuple[key_to_index["tw_end"]] if "tw_end" in key_to_index else None

        # Ensure indices are valid (adjust 1-based to 0-based)
        num_original_nodes = len(original_node_xy)
        subproblem_indices_0based = []
        for idx_1based in subproblem_node_indices:
            if 1 <= idx_1based <= num_original_nodes:
                subproblem_indices_0based.append(idx_1based - 1)
            else:
                logger.warning(f"create_subproblem_instance: Invalid node index {idx_1based} found (max is {num_original_nodes}). Skipping.")
                # Depending on desired behavior, could return None here instead of skipping

        if not subproblem_indices_0based:
            logger.error(f"create_subproblem_instance: No valid node indices left after validation for subproblem: {subproblem_node_indices}")
            return None

        # Select data for the subproblem nodes
        subproblem_node_xy = [original_node_xy[i] for i in subproblem_indices_0based]
        subproblem_demand = [original_demand[i] for i in subproblem_indices_0based]

        subproblem_data = {
            "depot_xy": depot_xy,
            "node_xy": subproblem_node_xy,
            "demand": subproblem_demand,
            "capacity": capacity
        }

        # Add optional fields if they exist for the problem type
        if "route_limit" in format_keys:
            subproblem_data["route_limit"] = route_limit
        if "service_time" in format_keys and original_service_time is not None:
            subproblem_data["service_time"] = [original_service_time[i] for i in subproblem_indices_0based]
        if "tw_start" in format_keys and original_tw_start is not None:
            subproblem_data["tw_start"] = [original_tw_start[i] for i in subproblem_indices_0based]
        if "tw_end" in format_keys and original_tw_end is not None:
            subproblem_data["tw_end"] = [original_tw_end[i] for i in subproblem_indices_0based]

        # Construct the final tuple in the correct order
        final_subproblem_tuple = tuple(subproblem_data[key] for key in format_keys)

        return final_subproblem_tuple

    except IndexError as e:
        logger.error(f"create_subproblem_instance: Error accessing data for {problem_type} at expected indices. Check VRP_DATA_FORMAT and instance tuple structure. Error: {e}")
        return None
    except KeyError as e:
         logger.error(f"create_subproblem_instance: Missing expected key '{e}' in VRP_DATA_FORMAT or data structure for {problem_type}.")
         return None
    except Exception as e:
        logger.exception(f"create_subproblem_instance: Unexpected error creating subproblem instance: {e}")
        return None


def _pad_one_instance_to_target(instance_data, problem_type, target_node_count, large_time_value=99999.0):
    """(Helper) Pads or truncates a single VRP instance tuple to a target node count.

    Args:
        instance_data (tuple): The raw instance tuple to pad/truncate.
        problem_type (str): The type of VRP problem (e.g., "CVRP", "VRPTW").
        target_node_count (int): The desired number of customer nodes.
        large_time_value (float): Value to use for the end time of padding nodes' TW.

    Returns:
        tuple or None: The modified instance tuple, or None if problem_type is invalid.
                       Logs error if truncation occurs.
    """
    if problem_type not in VRP_DATA_FORMAT:
        logger.error(f"_pad_one_instance_to_target: Invalid problem_type: {problem_type}")
        return None

    format_keys = VRP_DATA_FORMAT[problem_type]
    key_to_index = {key: idx for idx, key in enumerate(format_keys)}

    # Mutable copy of the original data lists/values
    data_dict = {key: list(instance_data[idx]) if isinstance(instance_data[idx], (list, tuple)) else instance_data[idx] 
                 for key, idx in key_to_index.items()}

    current_nodes = len(data_dict.get("node_xy", []))
    num_change = target_node_count - current_nodes

    if num_change == 0:
        return instance_data # No change needed

    depot_coord_for_padding = data_dict["depot_xy"][0] # Use the first depot coordinate

    if num_change > 0: # --- Padding --- 
        padding_coords = [depot_coord_for_padding] * num_change
        padding_demands = [0.0] * num_change
        padding_service_time = [0.0] * num_change
        padding_tw_start = [0.0] * num_change
        padding_tw_end = [large_time_value] * num_change

        data_dict["node_xy"].extend(padding_coords)
        data_dict["demand"].extend(padding_demands)

        if "service_time" in data_dict and data_dict["service_time"] is not None:
            data_dict["service_time"].extend(padding_service_time)
        if "tw_start" in data_dict and data_dict["tw_start"] is not None:
            data_dict["tw_start"].extend(padding_tw_start)
        if "tw_end" in data_dict and data_dict["tw_end"] is not None:
            data_dict["tw_end"].extend(padding_tw_end)

    elif num_change < 0: # --- Truncation --- 
        num_to_keep = target_node_count
        logger.warning(f"_pad_one_instance_to_target: Instance with {current_nodes} nodes is larger than target {target_node_count}. Truncating.")
        data_dict["node_xy"] = data_dict["node_xy"][:num_to_keep]
        data_dict["demand"] = data_dict["demand"][:num_to_keep]

        if "service_time" in data_dict and data_dict["service_time"] is not None:
            data_dict["service_time"] = data_dict["service_time"][:num_to_keep]
        if "tw_start" in data_dict and data_dict["tw_start"] is not None:
            data_dict["tw_start"] = data_dict["tw_start"][:num_to_keep]
        if "tw_end" in data_dict and data_dict["tw_end"] is not None:
            data_dict["tw_end"] = data_dict["tw_end"][:num_to_keep]

    # Reconstruct padded/truncated instance tuple in correct order
    final_tuple = tuple(data_dict[key] for key in format_keys)
    return final_tuple


def pad_subproblem_batch(subproblem_instance_tuples, problem_type, target_pad_size: int | None = None):
    """Pads a batch of subproblem instances.

    Defaults to padding to the size of the largest instance in the batch.
    If target_pad_size is provided, pads all instances to that size.

    Args:
        subproblem_instance_tuples (list[tuple]): List of subproblem VRP instance tuples.
        problem_type (str): The VRP type of the subproblems.
        target_pad_size (int | None, optional): The target size to pad all instances to.
                                                  If None, pads to the max nodes found in the batch.
                                                  Defaults to None.

    Returns:
        tuple[list[tuple], int] or tuple[list, int]:
            - List of padded VRP instance tuples.
            - The actual padding size used for this batch.
            Returns ([], 0) if input list is empty or error occurs.
    """
    if not subproblem_instance_tuples:
        return [], 0

    if problem_type not in VRP_DATA_FORMAT:
        logger.error(f"pad_subproblem_batch: Invalid problem_type: {problem_type}")
        return [], 0

    key_to_index = {key: idx for idx, key in enumerate(VRP_DATA_FORMAT[problem_type])}
    node_xy_idx = key_to_index.get("node_xy")

    if node_xy_idx is None:
         logger.error(f"pad_subproblem_batch: Could not find 'node_xy' index for problem type {problem_type}.")
         return [], 0

    # Find max nodes in the current batch
    max_nodes_in_batch = 0
    for inst_tuple in subproblem_instance_tuples:
        try:
            # Assuming node_xy includes depot, so length is num_nodes + 1
            # Adjust if node_xy only contains customer nodes
            num_nodes = len(inst_tuple[node_xy_idx])
            max_nodes_in_batch = max(max_nodes_in_batch, num_nodes)
        except (IndexError, TypeError, AttributeError):
             logger.warning("pad_subproblem_batch: Error accessing node_xy list in an instance tuple. Skipping this instance for max node calculation.", exc_info=True)
             continue # Skip malformed tuples for max calculation

    # Determine the final padding size
    actual_pad_size: int
    if target_pad_size is not None:
        actual_pad_size = target_pad_size
        logger.debug(f"Padding subproblem batch using provided target size: {actual_pad_size}")
    else:
        # Default behavior: pad to max nodes in batch (at least 1)
        if max_nodes_in_batch == 0:
             logger.warning("pad_subproblem_batch: Max nodes in batch is 0. Padding to size 1.")
             actual_pad_size = 1
        else:
             actual_pad_size = max_nodes_in_batch
        logger.debug(f"Padding subproblem batch using calculated max nodes: {actual_pad_size}")


    padded_batch = []
    for inst_tuple in subproblem_instance_tuples:
        # Pad to the determined actual_pad_size
        padded_instance = _pad_one_instance_to_target(inst_tuple, problem_type, actual_pad_size)
        if padded_instance:
             padded_batch.append(padded_instance)
        else:
             logger.error("pad_subproblem_batch: Failed to pad one instance. Skipping it.")
             # Potentially return [], 0 immediately if strict error handling is needed

    if len(padded_batch) != len(subproblem_instance_tuples):
         logger.warning("pad_subproblem_batch: Some instances failed padding and were skipped.")

    return padded_batch, actual_pad_size # Return the actually used padding size


def remove_padding_zeros(path_list):
    """Removes leading/trailing zeros from a path sequence."""
    start = 0
    while start < len(path_list) and path_list[start] == 0:
        start += 1
    end = len(path_list) - 1
    while end >= 0 and path_list[end] == 0:
        end -= 1
    return path_list[start:end+1] if start <= end else []


def solve_vrp_batch(solver_model, solver_env_class, original_instance_tuples, padded_batch_data, padded_problem_size, problem_type, device, aug_factor=8):
    """Solves a batch of pre-processed and padded VRP instances using the solver model.

    Args:
        solver_model (torch.nn.Module): The loaded and initialized solver MoE model.
        solver_env_class (type): The environment class corresponding to the problem_type
                                 (e.g., CVRPEnv, VRPTWEnv).
        original_instance_tuples (list[tuple]): List of the original, unpadded VRP instance tuples.
                                                 Used for calculating realistic pomo_size.
        padded_batch_data (tuple): A tuple of batched tensors (depot_xy, node_xy, node_demand, ...)
                                   representing the padded subproblem instances. Depot/node coordinates
                                   should be original scale, demand should be normalized by capacity.
        padded_problem_size (int): The number of nodes (including padding) in the instances.
        problem_type (str): The original VRP type (used for env initialization).
        device (torch.device): The device for computation.
        aug_factor (int, optional): Augmentation factor for solving. Defaults to 8.

    Returns:
        list[tuple]: A list of results, one tuple per input instance.
                     Each tuple contains (aug_score, best_aug_path), where path is optional
                     and may be None if unavailable. Returns empty list on error.
    """
    if not padded_batch_data or not original_instance_tuples or solver_env_class is None or solver_model is None:
        logger.error("solve_vrp_batch: Invalid input - missing model, env_class, original data, or padded data.")
        return []

    try:
        batch_size = padded_batch_data[0].size(0)
        if batch_size == 0 or batch_size != len(original_instance_tuples):
            logger.error(f"solve_vrp_batch: Batch size mismatch or zero batch size (Padded: {batch_size}, Original: {len(original_instance_tuples)})")
            return []

        # --- Calculate Realistic Pomo Size based on the first original instance --- 
        first_orig_tuple = original_instance_tuples[0]
        key_to_index = {key: idx for idx, key in enumerate(VRP_DATA_FORMAT.get(problem_type, []))}
        orig_node_xy_idx = key_to_index.get("node_xy")
        orig_demand_idx = key_to_index.get("demand")
        
        calculated_pomo_size = padded_problem_size # Default fallback
        if orig_node_xy_idx is not None and orig_demand_idx is not None and len(first_orig_tuple) > max(orig_node_xy_idx, orig_demand_idx):
            original_node_xy = first_orig_tuple[orig_node_xy_idx]
            original_demand = first_orig_tuple[orig_demand_idx]
            effective_problem_size = len(original_node_xy)
            
            if effective_problem_size > 0:
                # Count nodes with non-zero demand as potential starting points
                num_linehaul_nodes = sum(1 for d in original_demand if d != 0) 
                
                # Apply VRPB logic if applicable
                pomo_size_limit = effective_problem_size
                if 'B' in problem_type: 
                    backhaul_ratio = 0.2 # Assume default, could be made configurable
                    pomo_size_limit = int(effective_problem_size * (1 - backhaul_ratio))
                
                calculated_pomo_size = min(num_linehaul_nodes, pomo_size_limit)
                calculated_pomo_size = max(1, calculated_pomo_size) # Ensure at least 1
                calculated_pomo_size = min(calculated_pomo_size, effective_problem_size) # Cannot exceed actual nodes
            else:
                calculated_pomo_size = 1 # If no effective nodes, pomo is 1
            # logger.info(f"Calculated pomo size based on original instance: {calculated_pomo_size} (Effective: {effective_problem_size}, Linehaul: {num_linehaul_nodes}, Limit: {pomo_size_limit if 'B' in problem_type else 'N/A'})")
        else:
            logger.warning("Could not extract original data to calculate pomo size accurately. Using padded_problem_size as fallback.")
            calculated_pomo_size = max(1, padded_problem_size) # Fallback to padded size
        
        # --- Environment Initialization --- 
        env_params = {"problem_size": padded_problem_size, "pomo_size": calculated_pomo_size, "device": device} # Use calculated pomo, add device
        # Add route_limit if applicable (needs to be passed or inferred)
        # Assuming route_limit is part of padded_batch_data if needed by env
        if problem_type in VRP_DATA_FORMAT:
             format_keys = VRP_DATA_FORMAT[problem_type]
             if "route_limit" in format_keys:
                  rl_index = format_keys.index("route_limit") 
                  # Check index relative to the *tensor* tuple length
                  tensor_tuple_keys = [k for k in format_keys if k != 'capacity']
                  try:
                      tensor_rl_index = tensor_tuple_keys.index("route_limit")
                      if len(padded_batch_data) > tensor_rl_index:
                          route_limit_scalar = padded_batch_data[tensor_rl_index][0, 0].item()
                          env_params["route_limit"] = route_limit_scalar
                          logger.debug(f"Using route_limit={route_limit_scalar} for environment.")
                      else:
                           logger.warning(f"route_limit expected but tensor tuple too short (idx {tensor_rl_index} vs len {len(padded_batch_data)}).")
                  except ValueError:
                      logger.warning("Could not find route_limit in tensor keys after excluding capacity.")

        env = solver_env_class(**env_params)
        effective_env_pomo_size = env.pomo_size # Get the size the env will actually use
        logger.debug(f"Solver Env initialized. Padded Size: {padded_problem_size}, Requested Pomo: {calculated_pomo_size}, Effective Pomo: {effective_env_pomo_size}")

        # --- Batched Inference --- 
        solver_model.eval() # Ensure solver is in eval mode
        results = []

        with torch.no_grad():
            # Load problems into environment
            env.load_problems(batch_size, problems=padded_batch_data, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()

            # Pre-forward step
            solver_model.pre_forward(reset_state) 

            # Rollout loop
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = solver_model(state) 
                state, reward, done = env.step(selected)

            # --- Process Results --- 
            # Get the effective pomo size the environment *actually* used for this batch
            # This might differ from the initial calculation if the env adjusted it (e.g., VRPB)
            current_pomo_size = env.pomo_size
            expected_elements_from_env_pomo = batch_size * aug_factor * current_pomo_size
            actual_elements = reward.numel()

            # Check if the returned reward tensor size matches what we expect based on env.pomo_size
            if actual_elements != expected_elements_from_env_pomo:
                # If it doesn't match, try to infer the pomo size from the actual elements
                # This assumes batch_size and aug_factor are correct.
                if actual_elements % (batch_size * aug_factor) == 0:
                    inferred_pomo_size = actual_elements // (batch_size * aug_factor)
                    logger.warning(
                        f"Reward tensor size mismatch! Expected {expected_elements_from_env_pomo} elements based on env.pomo_size={current_pomo_size}, "
                        f"but got {actual_elements}. Attempting reshape with inferred pomo_size={inferred_pomo_size}."
                    )
                    # Use the inferred size for reshaping reward AND path tensors later
                    current_pomo_size = inferred_pomo_size
                else:
                    # Cannot safely infer pomo size, dimensions are incompatible.
                    logger.error(
                        f"Reward tensor size mismatch AND incompatible! Expected {expected_elements_from_env_pomo} elements (B={batch_size}, Aug={aug_factor}, EnvPomo={env.pomo_size}), "
                        f"but got {actual_elements}. Cannot infer pomo size. Cannot reshape safely."
                    )
                    return [(float('inf'), None)] * batch_size # Return failures

            # Reshape reward using the potentially corrected current_pomo_size
            try:
                aug_reward = reward.reshape(aug_factor, batch_size, current_pomo_size)
            except RuntimeError as reshape_err:
                 logger.error(f"Failed to reshape reward tensor with size {actual_elements} to ({aug_factor}, {batch_size}, {current_pomo_size}): {reshape_err}")
                 return [(float('inf'), None)] * batch_size # Return failures

            # Attempt to get paths
            selected_paths_tensor = None
            best_paths_unavailable = False
            if hasattr(env, 'selected_node_list') and env.selected_node_list is not None:
                try:
                    # Validate dimensions before reshaping
                    if env.selected_node_list.shape[0] != batch_size * aug_factor:
                         raise ValueError(f"Unexpected first dimension for selected_node_list: {env.selected_node_list.shape[0]} vs {batch_size * aug_factor}")
                    if env.selected_node_list.shape[1] != current_pomo_size:
                         logger.warning(f"Pomo dimension mismatch in selected_node_list ({env.selected_node_list.shape[1]}) vs env ({current_pomo_size}). Using list's dim.")
                         current_pomo_size = env.selected_node_list.shape[1]
                         aug_reward = reward.reshape(aug_factor, batch_size, current_pomo_size) # Re-reshape reward if pomo changed

                    seq_len = env.selected_node_list.shape[-1]
                    selected_paths_tensor = env.selected_node_list.reshape(aug_factor, batch_size, current_pomo_size, seq_len)
                except Exception as reshape_err:
                     logger.warning(f"Could not reshape selected_node_list: {reshape_err}. Paths will be unavailable.")
                     best_paths_unavailable = True
            else:
                logger.warning("env.selected_node_list not found or is None. Paths will be unavailable.")
                best_paths_unavailable = True

            # Extract results for each instance in the batch
            for i in range(batch_size):
                aug_reward_i = aug_reward[:, i, :] # shape: [aug, current_pomo_size]

                # Calculate augmented score
                if current_pomo_size > 0:
                    max_pomo_reward_per_aug_i, _ = aug_reward_i.max(dim=1) # shape: [aug]
                    max_aug_pomo_reward_i, best_aug_idx_tensor_i = max_pomo_reward_per_aug_i.max(dim=0)
                    aug_score_i = -max_aug_pomo_reward_i.item() # Use item() for scalar tensor
                    best_aug_idx_i = best_aug_idx_tensor_i.item()
                else: # Handle case where pomo size ended up as 0
                    aug_score_i = float('inf') # Or some other indicator of failure
                    best_aug_idx_i = 0
                    logger.warning(f"Instance {i}: Pomo size is zero, cannot calculate score.")


                # Get best augmented path (optional)
                best_aug_path_i = None
                if not best_paths_unavailable and current_pomo_size > 0:
                    try:
                        best_pomo_idx_within_best_aug_i = aug_reward_i[best_aug_idx_i, :].argmax().item()
                        raw_path = selected_paths_tensor[best_aug_idx_i, i, best_pomo_idx_within_best_aug_i, :].cpu().tolist()
                        best_aug_path_i = remove_padding_zeros(raw_path)
                    except Exception as path_err:
                        logger.warning(f"Error extracting path for instance {i}: {path_err}")
                        best_aug_path_i = None # Set path to None if error occurs


                results.append((aug_score_i, best_aug_path_i))

        return results

    except Exception as e:
        logger.exception(f"Error during solve_vrp_batch execution: {e}")
        # Return list of failures matching batch size for graceful handling downstream
        return [(float('inf'), None)] * batch_size


def prepare_batch_tensor_data(padded_instance_tuples, problem_type, device):
    """Converts a list of padded instance tuples into a tuple of batched tensors.

    Args:
        padded_instance_tuples (list[tuple]): List of padded VRP instance tuples.
                                                All tuples must have the same structure and padded size.
        problem_type (str): The VRP problem type.
        device (torch.device): The target device for the tensors.

    Returns:
        tuple or None: A tuple containing batched tensors (depot_xy, node_xy, demand, ...)
                       in the order defined by VRP_DATA_FORMAT[problem_type].
                       Returns None if input is invalid or conversion fails.
    """
    if not padded_instance_tuples:
        return None
    if problem_type not in VRP_DATA_FORMAT:
        logger.error(f"prepare_batch_tensor_data: Unknown problem_type: {problem_type}")
        return None

    format_keys = VRP_DATA_FORMAT[problem_type]
    key_to_index = {key: idx for idx, key in enumerate(format_keys)}
    batch_size = len(padded_instance_tuples)

    # Initialize lists to hold data for each component before stacking
    batch_data_lists = {key: [] for key in format_keys}
    # Check expected length based on first instance (assuming consistency)
    expected_len = len(padded_instance_tuples[0])
    if expected_len < len(format_keys):
        logger.error(f"prepare_batch_tensor_data: First instance tuple length ({expected_len}) is less than expected ({len(format_keys)}) for {problem_type}.")
        return None

    # --- Extract data and check consistency --- 
    first_node_count = -1
    for i, instance_tuple in enumerate(padded_instance_tuples):
        if len(instance_tuple) != expected_len:
            logger.error(f"prepare_batch_tensor_data: Instance {i} has length {len(instance_tuple)}, expected {expected_len}. Skipping batch.")
            return None

        # Check node count consistency (important after padding)
        current_node_count = len(instance_tuple[key_to_index['node_xy']])
        if first_node_count == -1:
            first_node_count = current_node_count
        elif current_node_count != first_node_count:
            logger.error(f"prepare_batch_tensor_data: Inconsistent node counts after padding found in batch (instance {i}: {current_node_count}, expected: {first_node_count}). Skipping batch.")
            return None

        # Append data to lists
        for key, idx in key_to_index.items():
            batch_data_lists[key].append(instance_tuple[idx])

    # --- Convert lists to tensors --- 
    batch_tensors = {}
    try:
        # Special handling for depot_xy (needs unsqueeze and repeat if needed, but assume solver handles B=1 case)
        # Here we just stack it: (B, 1, 2)
        batch_tensors["depot_xy"] = torch.tensor(batch_data_lists["depot_xy"], dtype=torch.float32, device=device)
        # Node related data (B, N, Dims) or (B, N)
        batch_tensors["node_xy"] = torch.tensor(batch_data_lists["node_xy"], dtype=torch.float32, device=device)
        # Normalize demand by capacity before converting to tensor
        capacity_list = batch_data_lists["capacity"]
        demand_list_of_lists = batch_data_lists["demand"]
        normalized_demand = []
        for j in range(batch_size):
            cap = capacity_list[j]
            demands = demand_list_of_lists[j]
            if cap <= 0:
                 logger.warning(f"Instance {j} has non-positive capacity ({cap}). Using raw demands.")
                 normalized_demand.append(demands)
            else:
                 normalized_demand.append([d / cap for d in demands])
        batch_tensors["demand"] = torch.tensor(normalized_demand, dtype=torch.float32, device=device)

        # Optional fields
        if "capacity" in batch_tensors: del batch_data_lists["capacity"] # Capacity is used for normalization, not usually passed as tensor

        if "route_limit" in format_keys:
            # Assume route_limit is scalar, stack into [B, 1]
            batch_tensors["route_limit"] = torch.tensor(batch_data_lists["route_limit"], dtype=torch.float32, device=device).unsqueeze(-1)
        if "service_time" in format_keys:
            batch_tensors["service_time"] = torch.tensor(batch_data_lists["service_time"], dtype=torch.float32, device=device)
        if "tw_start" in format_keys:
            batch_tensors["tw_start"] = torch.tensor(batch_data_lists["tw_start"], dtype=torch.float32, device=device)
        if "tw_end" in format_keys:
            batch_tensors["tw_end"] = torch.tensor(batch_data_lists["tw_end"], dtype=torch.float32, device=device)

    except (TypeError, ValueError, IndexError) as e:
        logger.exception(f"prepare_batch_tensor_data: Error converting data to tensors: {e}")
        return None

    # Construct the final tuple in the correct order
    try:
        final_tensor_tuple = tuple(batch_tensors[key] for key in format_keys if key != 'capacity') # Exclude capacity from final tensor tuple
        return final_tensor_tuple
    except KeyError as e:
        logger.error(f"prepare_batch_tensor_data: Missing key '{e}' in generated tensors when creating final tuple.")
        return None


def _split_sequence_by_zeros(raw_sequence):
    """Splits a raw sequence by zeros into initial subproblems."""
    if not raw_sequence:
        return []
    initial_subproblems = []
    current_subproblem = []
    for node_index in raw_sequence:
        if node_index == 0:
            if current_subproblem:
                initial_subproblems.append(list(current_subproblem))
                current_subproblem = []
        elif node_index > 0:
            current_subproblem.append(node_index)
    if current_subproblem:
        initial_subproblems.append(list(current_subproblem))
    return initial_subproblems

def _calculate_centroid(node_indices, original_loc):
    """Calculates the centroid of a list of nodes.

    Args:
        node_indices (list[int]): List of 1-based node indices.
        original_loc (list[list[float]]): List of [x, y] coordinates for all original nodes.

    Returns:
        tuple[np.ndarray | None, int]: The [x, y] centroid and the count of valid nodes used, 
                                       or (None, 0) if indices are invalid/empty.
    """
    if not node_indices:
        return None, 0
    
    # Convert to NumPy array for easier indexing if not already
    original_loc_np = np.array(original_loc) 
    
    # Ensure indices are valid 0-based integers
    valid_indices_0based = []
    num_original_nodes = len(original_loc_np)
    for idx_1based in node_indices:
        idx_0based = idx_1based - 1
        if 0 <= idx_0based < num_original_nodes:
            valid_indices_0based.append(idx_0based)
        else:
            # Use logger if available, otherwise print warning
            log_func = getattr(logger, 'warning', print) 
            log_func(f"_calculate_centroid: Invalid node index {idx_1based} found.")
            # Decide how to handle: skip, return None, etc. Let's skip.
    
    num_valid_nodes = len(valid_indices_0based)
    if num_valid_nodes == 0:
        return None, 0
        
    # Use NumPy array indexing for efficiency
    coords = original_loc_np[valid_indices_0based]
    centroid = np.mean(coords, axis=0)
    return centroid, num_valid_nodes


def merge_subproblems_by_centroid_fixed_size(
    initial_subproblems,
    original_loc,
    original_depot,
    problem_size_for_dynamic_target: int,
    merge_num: int = -1,
    target_node_count: int = 0,
    adaptive_tolerance: float = 0.2,
    problem_type: str = "CVRP"  # Add problem_type parameter with default value
    ):
    """Merges initial subproblems based on centroid proximity or adaptively by node count.

    Finds the largest angular gap between centroids (relative to depot) and starts
    the merging process after that gap, wrapping around circularly.
    Then applies either fixed-number or adaptive-size merging to the reordered sequence.

    Args:
        initial_subproblems (list[list[int]]): List of initial subproblem node lists (1-based indices).
        original_loc (np.ndarray or list): Coordinates of all original customer nodes. Shape (num_cust, 2).
        original_depot (np.ndarray or list): Coordinates of the depot. Shape (1, 2) or (2,).
        problem_size_for_dynamic_target (int): The N value of the original problem, used for dynamic target selection.
        merge_num (int, optional): The target number of initial subproblems per group (fixed merging),
                                   or <= 0 to trigger adaptive merging. Defaults to -1 (adaptive).
        target_node_count (int, optional): Explicit target number of nodes for adaptive merging.
                                           If <= 0, dynamic selection based on problem_size_for_dynamic_target is used.
                                           Defaults to 0.
        adaptive_tolerance (float, optional): Fractional tolerance for adaptive merging. Defaults to 0.2.
        problem_type (str, optional): The type of VRP problem (e.g., "CVRP", "VRPTW"). Defaults to "CVRP".

    Returns:
        list[list[int]]: List of final merged subproblem node lists.
    """
    log_func_info = getattr(logger, 'info', print)
    log_func_debug = getattr(logger, 'debug', print)
    log_func_warning = getattr(logger, 'warning', print)

    if not initial_subproblems: return []

    original_loc_np = np.array(original_loc)
    original_depot_np = np.array(original_depot).flatten()
    subproblem_data = []
    for i, nodes in enumerate(initial_subproblems):
        if not nodes: continue
        centroid, num_nodes = _calculate_centroid(nodes, original_loc_np)
        if centroid is not None:
            rel_x = centroid[0] - original_depot_np[0]
            rel_y = centroid[1] - original_depot_np[1]
            Theta = math.atan2(rel_y, rel_x) # Angle in [-pi, pi]
            subproblem_data.append({ 'id': i, 'Theta': Theta, 'nodes': nodes, 'count': num_nodes })
        else: log_func_warning(f"Could not calculate centroid/count for initial subproblem {i}, skipping.")

    num_valid_subproblems = len(subproblem_data)
    if num_valid_subproblems == 0: return []
    # --- Handle cases with 1 or 0 subproblems --- 
    if num_valid_subproblems <= 1:
        log_func_info(f"Only {num_valid_subproblems} valid subproblem(s), no merging needed.")
        return [sp['nodes'] for sp in subproblem_data] # Return list of node lists

    # --- Sort Subproblems by Angle ---
    subproblem_data.sort(key=lambda x: x['Theta'])

    # --- Find Largest Angular Gap ---
    max_gap = -1.0
    max_gap_index = -1 # Index *before* the largest gap

    for i in range(num_valid_subproblems - 1):
        diff = subproblem_data[i+1]['Theta'] - subproblem_data[i]['Theta']
        if diff < 0: diff += 2 * math.pi 
        if diff > max_gap:
            max_gap = diff
            max_gap_index = i

    # Check wrap-around difference (between last and first)
    wrap_diff = (subproblem_data[0]['Theta'] + 2 * math.pi) - subproblem_data[num_valid_subproblems-1]['Theta']
    if wrap_diff < 0: wrap_diff += 2 * math.pi

    if wrap_diff > max_gap:
        max_gap = wrap_diff
        max_gap_index = num_valid_subproblems - 1

    # --- Reorder Subproblems Starting After the Largest Gap ---
    start_merge_index = (max_gap_index + 1) % num_valid_subproblems
    log_func_debug(f"Largest angular gap found after index {max_gap_index} (Theta={subproblem_data[max_gap_index]['Theta']:.2f}). Starting merge loop from index {start_merge_index}.")
    reordered_subproblems = subproblem_data[start_merge_index:] + subproblem_data[:start_merge_index]

    # --- Apply Merging Logic (Fixed or Adaptive) to the REORDERED list ---
    final_merged_subproblems = []
    data_to_merge = reordered_subproblems 

    if merge_num > 0:
        # Fixed Number Merging (Applied to reordered list)
        # log_func_info(f"Using FIXED merging (merge_num = {merge_num}) starting after largest angle gap.")
        num_groups = math.ceil(num_valid_subproblems / merge_num)
        for i in range(num_groups):
            start_idx_in_reordered = i * merge_num
            group_data = []
            for j in range(merge_num):
                 current_idx_in_reordered = (start_idx_in_reordered + j)
                 # Check if we exceed the total number of valid subproblems
                 if current_idx_in_reordered < num_valid_subproblems: 
                      group_data.append(data_to_merge[current_idx_in_reordered])
                 else:
                      break # Stop adding if we've exhausted the subproblems

            merged_nodes = []
            for data in group_data:
                merged_nodes.extend(data['nodes'])
            if merged_nodes:
                final_merged_subproblems.append(merged_nodes)
            # else: log_func_warning(f"Fixed merging group {i} (reordered) resulted in empty node list.") # Less noisy

    else:
        # Adaptive Size Merging (Applied to reordered list)
        effective_target_node_count = 0
        if target_node_count <= 0:
            # choose target size based on problem type and size
            targets = {
                'CVRP': {
                    200: 100,
                    500: 125,
                    1000: 150,
                    2000: 300,
                },
                'OVRP': {
                    200: 100,
                    500: 125,
                    1000: 150,
                    2000: 300,
                },
                'VRPB': {
                    200: 100,
                    500: 125,
                    1000: 200,
                    2000: 300,
                },
                'VRPL': {
                    200: 100,
                    500: 125,
                    1000: 150,
                    2000: 300,
                },
                'VRPTW': {
                    200: 20,
                    500: 50,
                    1000: 50,
                    2000: 100,
                }
            }
            # get the closest problem size
            predefined_sizes = np.array(list(targets.get(problem_type, {}).keys()))
            if len(predefined_sizes) > 0:
                closest_idx = (np.abs(predefined_sizes - problem_size_for_dynamic_target)).argmin()
                closest_size = predefined_sizes[closest_idx]
                effective_target_node_count = targets.get(problem_type, {}).get(closest_size, 50)
            else:
                effective_target_node_count = 50  # default value
            log_func_debug(f"ADAPTIVE merging (post-gap): For {problem_type} N={problem_size_for_dynamic_target}, selected target = {effective_target_node_count}.")
        else:
            effective_target_node_count = target_node_count
            log_func_debug(f"ADAPTIVE merging (post-gap): Using explicit target = {effective_target_node_count}.")

        upper_bound = effective_target_node_count * (1 + adaptive_tolerance)
        processed_indices = set() # Track original indices ('id' field) to avoid double processing
        current_reordered_idx = 0 # Pointer into the data_to_merge list

        while len(processed_indices) < num_valid_subproblems:
            current_group_nodes = []
            current_group_count = 0
            
            # Start a new group with the first available unprocessed subproblem
            group_started = False
            start_search_idx = current_reordered_idx
            while not group_started:
                sub_data = data_to_merge[current_reordered_idx]
                if sub_data['id'] not in processed_indices:
                    current_group_nodes.extend(sub_data['nodes'])
                    current_group_count += sub_data['count']
                    processed_indices.add(sub_data['id'])
                    group_started = True
                    # Don't advance current_reordered_idx yet, let the inner loop do it
                else:
                    current_reordered_idx = (current_reordered_idx + 1) % num_valid_subproblems
                    if current_reordered_idx == start_search_idx:
                        log_func_warning("Adaptive merge error: Cycled without finding unprocessed start node.")
                        break # Break inner search loop
            if not group_started: break # Break outer while loop if no start found
            
            # Try adding subsequent unprocessed subproblems
            next_idx_to_consider = (current_reordered_idx + 1) % num_valid_subproblems
            steps_taken_in_group = 0 # Safety break
            while len(processed_indices) < num_valid_subproblems and steps_taken_in_group < num_valid_subproblems:
                next_sub_data = data_to_merge[next_idx_to_consider]
                
                # Only consider if not already processed
                if next_sub_data['id'] in processed_indices:
                    next_idx_to_consider = (next_idx_to_consider + 1) % num_valid_subproblems
                    steps_taken_in_group += 1
                    continue

                next_sub_count = next_sub_data['count']
                # Check if adding exceeds bound (only break if current group has nodes)
                if current_group_count + next_sub_count > upper_bound and current_group_count > 0:
                    break 
                else:
                    current_group_nodes.extend(next_sub_data['nodes'])
                    current_group_count += next_sub_count
                    processed_indices.add(next_sub_data['id'])
                    next_idx_to_consider = (next_idx_to_consider + 1) % num_valid_subproblems
                steps_taken_in_group += 1
            
            if current_group_nodes:
                final_merged_subproblems.append(current_group_nodes)
            # else: log_func_warning("Adaptive merging produced an empty group.") # Less noisy
            
            # Advance the main pointer for the next group's start search
            current_reordered_idx = next_idx_to_consider 

    # log_func_info(f"Merging (post-gap) resulted in {len(final_merged_subproblems)} final subproblems.")
    return final_merged_subproblems


def partition_instance(original_instance_tuple, problem_type,
                       partitioner_checkpoint_path, merge_num, device,
                       partitioner_model_params, # Params for the fine-tuned SOLVER model
                       max_seq_len_factor=2,
                       partitioner_model=None, # <<< Add optional pre-loaded model
                       target_node_count_for_merge: int = 0): # <<< Use 0 as default
    """
    Partitions a single VRP instance using a fine-tuned Solver model.

    Args:
        original_instance_tuple (tuple): The raw VRP instance data.
        problem_type (str): The type of VRP problem (e.g., "CVRP").
        partitioner_checkpoint_path (str): Path to the trained partitioner checkpoint
                                           (containing the fine-tuned Solver model state).
                                           Required if partitioner_model is None.
        merge_num (int): Number of consecutive raw subproblems to merge (fixed merging),
                         or <= 0 for adaptive merging.
        device (torch.device): Computation device.
        partitioner_model_params (dict): Parameters needed to initialize the Solver model
                                         architecture used for partitioning.
                                         Required if partitioner_model is None.
        max_seq_len_factor (int): Factor to multiply instance size by to get max sequence length.
        partitioner_model (torch.nn.Module, optional): Pre-loaded partitioner model.
                                                       If provided, loading from checkpoint is skipped.
        target_node_count_for_merge (int, optional): Target node count for adaptive merging.
                                                     <=0 triggers dynamic selection. Defaults to 0.

    Returns:
        tuple(list[tuple], list[int]) | tuple(None, None):
            - List of subproblem instance tuples.
            - Raw partition sequence generated by the model.
            Returns (None, None) on failure.
    """
    log_func_info = getattr(logger, 'info', print) # Use logger if available
    log_func_error = getattr(logger, 'error', print)
    log_func_warning = getattr(logger, 'warning', print)
    log_func_exception = getattr(logger, 'exception', print)

    # <<< Check if model is already loaded >>>
    if partitioner_model is None:
        # log_func_info(f"Partitioning instance (type: {problem_type}) using fine-tuned model: {partitioner_checkpoint_path}")
        # --- 1. Load the Fine-tuned Solver Model ---
        partitioner_model = load_moe_model( # Use generic loader now
            partitioner_checkpoint_path,
            device,
            model_type=partitioner_model_params.get('model_type', None), # Pass type if known
            model_params=partitioner_model_params
        )
        if not partitioner_model:
            log_func_error("Failed to load the fine-tuned model from checkpoint.")
            return None, None
        log_func_info("Partitioner model loaded from checkpoint.")
    else:
         # log_func_info(f"Partitioning instance (type: {problem_type}) using pre-loaded model.")
         # Ensure the pre-loaded model is on the correct device
         try:
              partitioner_model = partitioner_model.to(device)
              partitioner_model.eval() # Ensure it's in eval mode
         except Exception as e:
              log_func_error(f"Failed to move pre-loaded partitioner model to device {device} or set to eval: {e}")
              return None, None

    # --- 2. Determine Instance Size and Max Sequence Length --- 
    try:
        # Determine problem size (N) from the instance itself
        if problem_type not in VRP_DATA_FORMAT or len(original_instance_tuple) <= VRP_DATA_FORMAT[problem_type].index('node_xy'):
            raise ValueError("Cannot determine node_xy index from VRP_DATA_FORMAT or instance tuple is too short.")
        node_xy_index = VRP_DATA_FORMAT[problem_type].index('node_xy')
        num_customer_nodes = len(original_instance_tuple[node_xy_index]) 

        original_loc = np.array(original_instance_tuple[node_xy_index]) # Ensure numpy array for loc
        depot_index = VRP_DATA_FORMAT[problem_type].index('depot_xy')
        original_depot = np.array(original_instance_tuple[depot_index]) # Ensure numpy array for depot
        if original_depot.ndim > 1: original_depot = original_depot.flatten()
        if num_customer_nodes <= 0: raise ValueError("Instance has no customer nodes.")
    except (IndexError, TypeError, ValueError) as e:
        log_func_error(f"Could not determine instance data/size from tuple: {e}")
        return None, None
    max_seq_len = max_seq_len_factor * (num_customer_nodes + 1)
    # log_func_info(f"Instance size N={num_customer_nodes}, Max sequence length set to {max_seq_len}")

    # --- 3. Setup Environment for Rollout --- 
    try:
        EnvClassList = get_env(problem_type)
        if not EnvClassList: raise ValueError(f"Could not get env class for {problem_type}")
        PartitionEnvClass = EnvClassList[0]
        env_params = {"problem_size": num_customer_nodes, "pomo_size": 1, "device": device} # Pass device
        partition_env = PartitionEnvClass(**env_params)
        # Pad instance to its own size for rollout (no actual padding needed here)
        padded_batch_tuples, target_pad_size = pad_subproblem_batch([original_instance_tuple], problem_type, num_customer_nodes)
        if not padded_batch_tuples or target_pad_size != num_customer_nodes: raise ValueError(f"Padding/Target size mismatch. Expected {num_customer_nodes}, got {target_pad_size}")
        instance_tensor_data = prepare_batch_tensor_data(padded_batch_tuples, problem_type, device)
        if not instance_tensor_data: raise ValueError("Failed to prepare instance tensor data.")
        partition_env.load_problems(batch_size=1, problems=instance_tensor_data, aug_factor=1)
    except Exception as e:
        log_func_exception(f"Error setting up environment for partitioning: {e}")
        return None, None

    # --- 4. Generate Partition Sequence via Rollout --- 
    raw_sequence = None
    try:
        partitioner_model.eval() 
        partitioner_model.set_eval_type('argmax')
        with torch.no_grad():
            reset_state, _, _ = partition_env.reset()
            partitioner_model.pre_forward(reset_state)
            state, _, done = partition_env.pre_step()
            step_count = 0
            while not done and step_count < max_seq_len:
                selected, _ = partitioner_model(state) 
                state, _, done = partition_env.step(selected)
                step_count += 1
            if step_count >= max_seq_len: log_func_warning(f"Sequence generation reached max length ({max_seq_len}).")
            if hasattr(partition_env, 'selected_node_list') and partition_env.selected_node_list is not None:
                 if partition_env.selected_node_list.numel() > 0: raw_sequence = partition_env.selected_node_list.view(-1).cpu().tolist()
                 else: log_func_warning("partition_env.selected_node_list is empty.")
            else: log_func_warning("partition_env.selected_node_list not found.")
        if raw_sequence is None:
            log_func_error("Failed to generate sequence from environment rollout.")
            return None, None 
        log_func_info(f"partition_instance: Problem {problem_type}, N={num_customer_nodes}, MergeNum={merge_num}, TargetMergeSize={target_node_count_for_merge}") # Log params
        log_func_info(f"partition_instance: Raw sequence (len={len(raw_sequence)}): {str(raw_sequence[:30]) + '...' if len(raw_sequence) > 30 else raw_sequence}") # Log raw sequence

    except Exception as e:
        log_func_exception(f"Error during partitioning model rollout: {e}")
        return None, None

    # --- 5. Process Sequence (Split and Merge using MODIFIED function) --- 
    try:
        initial_subproblems = _split_sequence_by_zeros(raw_sequence)
        log_func_info(f"partition_instance: Initial subproblems from raw_sequence: {len(initial_subproblems)}")
        if initial_subproblems:
            log_func_info(f"partition_instance: First 3 initial subproblems (node counts): {[len(sp) for sp in initial_subproblems[:3]]}")

        if not initial_subproblems:
            log_func_warning("Raw sequence resulted in no initial subproblems.")
            return [], raw_sequence 

        # <<< Call merge function passing the determined problem size >>>
        merged_node_lists = merge_subproblems_by_centroid_fixed_size(
            initial_subproblems=initial_subproblems,
            original_loc=original_loc,
            original_depot=original_depot,
            problem_size_for_dynamic_target=num_customer_nodes, # <<< Pass actual N
            merge_num=merge_num, # Pass the user's choice (-1 for adaptive)
            target_node_count=target_node_count_for_merge # Pass the target size (0 for dynamic)
        )
        log_func_info(f"partition_instance: Merged subproblem lists: {len(merged_node_lists)}")
        if merged_node_lists:
            log_func_info(f"partition_instance: First 3 merged subproblem lists (node counts): {[len(sp_list) for sp_list in merged_node_lists[:3]]}")

        if not merged_node_lists:
            log_func_warning(f"Merging resulted in no final subproblems.")
            return [], raw_sequence

        # --- 6. Create Final Subproblem Tuples --- 
        subproblem_instance_tuples = []
        for node_indices in merged_node_lists:
            valid_indices = [idx for idx in node_indices if 1 <= idx <= num_customer_nodes]
            if not valid_indices:
                 log_func_warning(f"Skipping empty/invalid merged list: {node_indices}")
                 continue
            sub_instance = create_subproblem_instance(original_instance_tuple, problem_type, valid_indices)
            if sub_instance:
                subproblem_instance_tuples.append(sub_instance)
            else:
                log_func_warning(f"Failed to create subproblem instance for indices: {valid_indices}")

        log_func_info(f"partition_instance: Successfully created {len(subproblem_instance_tuples)} final subproblem instances.") # Log final count
        return subproblem_instance_tuples, raw_sequence

    except Exception as e:
        log_func_exception(f"Error processing sequence/creating final subproblems: {e}")
        return None, raw_sequence


def merge_solved_instances(raw_instance_datas, solved_sols=None):
    """
    Merges multiple raw VRP instances and optionally their corresponding solutions
    into a single larger instance and combined solution.

    Assumes shared depot, capacity, and route_limit across input instances.
    Remaps node indices in the solution paths if solutions are provided.

    Args:
        raw_instance_datas (list): A list of raw python instance tuples. Each tuple follows the order:
                           (depot_xy, node_xy, demands, capacity, [route_limit?], [service_time?], [tw_start?], [tw_end?])
        solved_sols (list, optional): A list of (score, path) tuples corresponding to raw_instance_datas.
                                      If None, only instance merging is performed. Defaults to None.

    Returns:
        tuple: (merged_instance, merged_solution)
               - merged_instance: A single tuple representing the combined instance data.
               - merged_solution: A tuple (total_score, combined_path) if solved_sols was provided,
                                  otherwise None.
               Returns (None, None) if an error occurs during instance merging.
    """
    if not raw_instance_datas:
        print("Error: Input 'raw_instance_datas' list cannot be empty.")
        return None, None

    # Determine if solutions should be merged
    merge_solutions = solved_sols is not None

    if merge_solutions and len(raw_instance_datas) != len(solved_sols):
        print(f"Error: Mismatch in lengths: {len(raw_instance_datas)} instances vs {len(solved_sols)} solutions.")
        return None, None

    print(f"\n--- Merging {len(raw_instance_datas)} instances ---")
    if merge_solutions:
        print("--- Merging corresponding solutions ---")

    try:
        # --- Validate Consistency and Get Structure ---
        first_instance = raw_instance_datas[0]
        depot_shared = first_instance[0]
        capacity_shared = float(first_instance[3])
        first_len = len(first_instance)

        # Determine structure based on first instance length
        has_L = False
        has_TW = False
        expected_len = 4
        if first_len >= 5: # Check if route_limit might be present
             if isinstance(first_instance[4], (int, float)):
                 has_L = True
                 expected_len += 1
        if first_len >= expected_len + 3: # Check if TW might be present
             if (isinstance(first_instance[expected_len], list) and
                 isinstance(first_instance[expected_len+1], list) and
                 isinstance(first_instance[expected_len+2], list)):
                 has_TW = True
                 expected_len += 3

        print(f">> Detected structure: Has_L={has_L}, Has_TW={has_TW}, Expected Len={expected_len}")

        route_limit_shared = float(first_instance[4]) if has_L else None

        # Lists to accumulate merged data
        all_node_xy = []
        all_demands = []
        all_service_times = [] if has_TW else None
        all_tw_starts = [] if has_TW else None
        all_tw_ends = [] if has_TW else None

        # Validate consistency across all instances
        for i, instance_data in enumerate(raw_instance_datas):
            if len(instance_data) != expected_len:
                raise ValueError(f"Instance {i} has length {len(instance_data)}, but expected {expected_len} based on first instance.")
            if instance_data[0] != depot_shared:
                raise ValueError(f"Instance {i} depot {instance_data[0]} differs from shared depot {depot_shared}.")
            if float(instance_data[3]) != capacity_shared:
                raise ValueError(f"Instance {i} capacity {float(instance_data[3])} differs from shared capacity {capacity_shared}.")
            if has_L and float(instance_data[4]) != route_limit_shared:
                 raise ValueError(f"Instance {i} route_limit {float(instance_data[4])} differs from shared route_limit {route_limit_shared}.")

            # Accumulate node-specific data
            nodes = instance_data[1]
            demands = instance_data[2]
            if not nodes: raise ValueError(f"Instance {i} has an empty node list.")
            if len(nodes) != len(demands): raise ValueError(f"Instance {i}: Node count ({len(nodes)}) != demand count ({len(demands)}).")

            all_node_xy.extend(nodes)
            all_demands.extend(demands)

            if has_TW:
                st = instance_data[expected_len - 3]
                ts = instance_data[expected_len - 2]
                te = instance_data[expected_len - 1]
                if len(st) != len(nodes): raise ValueError(f"Instance {i}: Service time count ({len(st)}) != node count ({len(nodes)}).")
                if len(ts) != len(nodes): raise ValueError(f"Instance {i}: TW start count ({len(ts)}) != node count ({len(nodes)}).")
                if len(te) != len(nodes): raise ValueError(f"Instance {i}: TW end count ({len(te)}) != node count ({len(nodes)}).")
                all_service_times.extend(st)
                all_tw_starts.extend(ts)
                all_tw_ends.extend(te)

        # --- Construct Merged Instance Tuple ---
        merged_instance_list = [
            depot_shared,
            all_node_xy,
            all_demands,
            capacity_shared
        ]
        if has_L:
            merged_instance_list.append(route_limit_shared)
        if has_TW:
            merged_instance_list.extend([all_service_times, all_tw_starts, all_tw_ends])

        merged_instance = tuple(merged_instance_list)
        print(f">> Merged instance created with {len(all_node_xy)} total nodes.")

        # --- Merge Solutions (Only if provided) ---
        merged_solution = None
        if merge_solutions:
            total_score = 0
            combined_path = []
            node_offset = 0

            for i, (score, path) in enumerate(solved_sols):
                total_score += score
                if not isinstance(path, list):
                     print(f"Warning: Path for solution {i} is not a list ('{path}'). Skipping path merge for this instance.")
                     continue # Skip if path is unavailable or not a list

                remapped_segment = []
                num_nodes_in_instance = len(raw_instance_datas[i][1])

                # Remap node indices (depot 0 stays 0, others get offset)
                for node_idx in path:
                    if node_idx == 0:
                        remapped_segment.append(0)
                    elif node_idx > 0:
                        if node_idx > num_nodes_in_instance:
                             print(f"Warning: Path for solution {i} contains invalid node index {node_idx} (max expected: {num_nodes_in_instance}). Skipping this index.")
                             continue
                        remapped_segment.append(node_idx + node_offset)
                    else:
                         print(f"Warning: Path for solution {i} contains invalid index {node_idx}. Skipping.")
                         continue
                         
                # --- Add depot visit (0) between subproblem solutions --- 
                if i > 0 and combined_path and remapped_segment:
                    # Check if the previous path segment already ended with 0
                    # And if the current segment starts with 0. Avoid double zeros.
                    if combined_path[-1] != 0 and remapped_segment[0] != 0:
                        combined_path.append(0)
                    elif combined_path[-1] == 0 and remapped_segment[0] == 0:
                        # If both end/start with 0, remove one before extending
                        remapped_segment = remapped_segment[1:]
                
                # --- Append the remapped segment --- 
                if remapped_segment: # Ensure segment is not empty after potential modification
                    combined_path.extend(remapped_segment)
                
                node_offset += num_nodes_in_instance # Update offset for the next instance

            # --- Final check: Ensure path doesn't end with 0 if it's not empty --- 
            # (Optional, depending on how downstream processing expects the path)
            if combined_path and combined_path[-1] == 0:
                combined_path.pop() 

            merged_solution = (total_score, combined_path)
            print(f">> Merged solution created: Total Score={total_score:.4f}, Path Length={len(combined_path)}")

        return merged_instance, merged_solution # Return merged_solution (or None if not merged)

    except ValueError as e:
        print(f"Error during instance validation/merging: {e}")
        traceback.print_exc()
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during merging: {e}")
        traceback.print_exc()
        return None, None


# --- Main execution block for testing ---
if __name__ == '__main__':
    # Configure Logger for Testing
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s') # Simpler format

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    print("=" * 60)

    total_passed = 0
    total_failed = 0

    # --- Testing load_moe_model ---
    print("--- Testing load_moe_model ---")
    test_passed_count = 0
    test_failed_count = 0

    # Test Case 1
    print("Test 1.1: Infer MOE type, Default Params (mvmoe_4e_n100)")
    cp1 = r"pretrained/mvmoe_4e_n100/epoch-5000.pt"
    m1 = load_moe_model(cp1, device)
    if m1 and isinstance(m1, MOEModel):
        print("  Result: PASS - Loaded MOEModel")
        test_passed_count += 1
    else:
        print(f"  Result: FAIL - Loaded: {type(m1)}")
        test_failed_count += 1

    # Test Case 2
    print("Test 1.2: Infer MOE_LIGHT type, Default Params (mvmoe_4e_light_n50)")
    cp2 = r"pretrained/mvmoe_4e_light_n50/epoch-5000.pt"
    m2 = load_moe_model(cp2, device)
    if m2 and isinstance(m2, MOEModel_Light):
        print("  Result: PASS - Loaded MOEModel_Light")
        test_passed_count += 1
    else:
        print(f"  Result: FAIL - Loaded: {type(m2)}")
        test_failed_count += 1

    # Test Case 3
    print("Test 1.3: Provide MOE type and Params (mvmoe_8e_n50)")
    cp3 = r"pretrained/mvmoe_8e_n50/epoch-2500.pt"
    p3 = DEFAULT_MODEL_PARAMS.copy(); p3["num_experts"] = 8
    m3 = load_moe_model(cp3, device, model_type="MOE", model_params=p3)
    if m3 and isinstance(m3, MOEModel) and getattr(m3, 'model_params', {}).get('num_experts') == 8:
        print("  Result: PASS - Loaded MOEModel with 8 experts")
        test_passed_count += 1
    else:
        print(f"  Result: FAIL - Loaded: {type(m3)}, Experts: {getattr(m3, 'model_params', {}).get('num_experts', 'N/A')}")
        test_failed_count += 1
    print(f"load_moe_model Tests: {test_passed_count} Passed, {test_failed_count} Failed")
    total_passed += test_passed_count
    total_failed += test_failed_count
    print("=" * 60)
    del m1, m2, m3 # Cleanup models

    # --- Testing create_subproblem_instance ---
    print("--- Testing create_subproblem_instance ---")
    test_passed_count = 0
    test_failed_count = 0
    if not VRP_DATA_FORMAT:
        print("SKIPPING: VRP_DATA_FORMAT not loaded.")
    else:
        # Test 3.1: CVRP
        print("Test 3.1: CVRP subproblem [2, 4]")
        original_cvrp = ([[0.5, 0.5]], [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.6]], [10, 20, 15, 25, 18], 100.0)
        expected_cvrp = ([[0.5, 0.5]], [[0.2, 0.2], [0.4, 0.4]], [20, 25], 100.0)
        result_cvrp = create_subproblem_instance(original_cvrp, "CVRP", [2, 4])
        if result_cvrp == expected_cvrp:
            print(f"  Result: PASS") # Output: {result_cvrp})
            test_passed_count += 1
        else:
            print(f"  Result: FAIL - Output: {result_cvrp}")
            test_failed_count += 1

        # Test 3.2: VRPTW
        print("Test 3.2: VRPTW subproblem [1, 3]")
        original_vrptw = ([[0.0, 0.0]], [[1, 1], [2, 2], [3, 3]], [5, 6, 7], 50.0, [1, 1, 1], [10, 20, 30], [100, 120, 130])
        expected_vrptw = ([[0.0, 0.0]], [[1, 1], [3, 3]], [5, 7], 50.0, [1, 1], [10, 30], [100, 130])
        result_vrptw = create_subproblem_instance(original_vrptw, "VRPTW", [1, 3])
        if result_vrptw == expected_vrptw:
            print(f"  Result: PASS") # Output: {result_vrptw})
            test_passed_count += 1
        else:
            print(f"  Result: FAIL - Output: {result_vrptw}")
            test_failed_count += 1

        # Test 3.3: Invalid Index
        print("Test 3.3: CVRP invalid index [2, 6]")
        expected_invalid = ([[0.5, 0.5]], [[0.2, 0.2]], [20], 100.0) # Only node 2 remains
        result_invalid = create_subproblem_instance(original_cvrp, "CVRP", [2, 6])
        if result_invalid == expected_invalid:
            print(f"  Result: PASS") # Output: {result_invalid})
            test_passed_count += 1
        else:
            print(f"  Result: FAIL - Output: {result_invalid}")
            test_failed_count += 1

        # Test 3.4: Empty List
        print("Test 3.4: CVRP empty list []")
        result_empty = create_subproblem_instance(original_cvrp, "CVRP", [])
        if result_empty is None:
            print(f"  Result: PASS - Output: {result_empty}")
            test_passed_count += 1
        else:
            print(f"  Result: FAIL - Output: {result_empty}")
            test_failed_count += 1
    print(f"create_subproblem_instance Tests: {test_passed_count} Passed, {test_failed_count} Failed")
    total_passed += test_passed_count
    total_failed += test_failed_count
    print("=" * 60)

    # --- Testing Padding Functions ---
    print("--- Testing Padding Functions ---")
    test_passed_count = 0
    test_failed_count = 0
    if not VRP_DATA_FORMAT:
        print("SKIPPING: VRP_DATA_FORMAT not loaded.")
    else:
        solver_native_size = 50
        # Create instances
        original_cvrp_large = ([[0.5, 0.5]], [[i*0.01, i*0.01] for i in range(1, 60)], [1]*59, 100.0)
        subprob_cvrp_small = create_subproblem_instance(original_cvrp_large, "CVRP", list(range(1, 11))) # 10 nodes
        subprob_cvrp_medium = create_subproblem_instance(original_cvrp_large, "CVRP", list(range(1, 41))) # 40 nodes
        subprob_cvrp_large = create_subproblem_instance(original_cvrp_large, "CVRP", list(range(1, 56))) # 55 nodes
        subprob_cvrp_toolarge = create_subproblem_instance(original_cvrp_large, "CVRP", list(range(1, 60))) # 59 nodes
        original_vrptw_small_t = ([[0.0, 0.0]], [[1,1],[2,2]], [5,6], 50.0, [1,1], [10,20], [100,120])
        sub_instance_vrptw_t = create_subproblem_instance(original_vrptw_small_t,"VRPTW",[1,2]) # 2 nodes

        if not all([subprob_cvrp_small, subprob_cvrp_medium, subprob_cvrp_large, subprob_cvrp_toolarge, sub_instance_vrptw_t]):
            print("SKIPPING Padding Tests: Failed to create necessary subproblem instances.")
        else:
            # Test 4.1: Batch smaller than native
            print("Test 4.1: Batch smaller than native (Max 40 -> Target 50)")
            batch_small = [subprob_cvrp_small, subprob_cvrp_medium]
            padded_batch_1, target_size_1 = pad_subproblem_batch(batch_small, "CVRP")
            if target_size_1 == 50 and len(padded_batch_1) == 2 and len(padded_batch_1[0][1]) == 50 and len(padded_batch_1[1][1]) == 50:
                print(f"  Result: PASS - Target={target_size_1}, Nodes={[len(p[1]) for p in padded_batch_1]}")
                test_passed_count += 1
            else:
                print(f"  Result: FAIL - Target={target_size_1}, Nodes={[len(p[1]) for p in padded_batch_1] if padded_batch_1 else []}")
                test_failed_count += 1

            # Test 4.2: Batch larger than native
            print("Test 4.2: Batch larger than native (Max 55 -> Target 55)")
            batch_large = [subprob_cvrp_medium, subprob_cvrp_large]
            padded_batch_2, target_size_2 = pad_subproblem_batch(batch_large, "CVRP")
            if target_size_2 == 55 and len(padded_batch_2) == 2 and len(padded_batch_2[0][1]) == 55 and len(padded_batch_2[1][1]) == 55:
                print(f"  Result: PASS - Target={target_size_2}, Nodes={[len(p[1]) for p in padded_batch_2]}")
                test_passed_count += 1
            else:
                print(f"  Result: FAIL - Target={target_size_2}, Nodes={[len(p[1]) for p in padded_batch_2] if padded_batch_2 else []}")
                test_failed_count += 1

            # Test 4.3: Empty Batch
            print("Test 4.3: Empty Batch")
            padded_batch_3, target_size_3 = pad_subproblem_batch([], "CVRP")
            if target_size_3 == 0 and padded_batch_3 == []:
                print(f"  Result: PASS - Target={target_size_3}, Batch={padded_batch_3}")
                test_passed_count += 1
            else:
                print(f"  Result: FAIL - Target={target_size_3}, Batch={padded_batch_3}")
                test_failed_count += 1

            # Test 4.4: Truncation Test (using helper)
            print("Test 4.4: Truncation helper (59 -> Target 30)")
            truncated_instance = _pad_one_instance_to_target(subprob_cvrp_toolarge, "CVRP", 30)
            if truncated_instance and len(truncated_instance[1]) == 30:
                print(f"  Result: PASS - Nodes={len(truncated_instance[1])}")
                test_passed_count += 1
            else:
                print(f"  Result: FAIL - Nodes={len(truncated_instance[1]) if truncated_instance else 'None'}")
                test_failed_count += 1
    print(f"Padding Function Tests: {test_passed_count} Passed, {test_failed_count} Failed")
    total_passed += test_passed_count
    total_failed += test_failed_count
    print("=" * 60)

    # --- Testing solve_vrp_batch ---
    print("--- Testing solve_vrp_batch ---")
    test_passed_count = 0
    test_failed_count = 0
    solver_model_instance = None # Keep loaded model for reuse

    # --- Test 5.1: OVRPBLTW ---
    print("Test 5.1: OVRPBLTW instance 0")
    # --- Specific Instance Setup ---
    instance_problem_type = "OVRPBLTW"
    dataset_base = "data" # Assuming 'data' folder is accessible
    dataset_path = os.path.join(dataset_base, 'OVRPBLTW', 'ovrpbltw50_uniform.pkl')
    instance_number = 0
    solution_path = os.path.join(dataset_base, 'OVRPBLTW', 'or_tools_20s_ovrpbltw50_uniform.pkl')
    solver_checkpoint_path = r"pretrained/mvmoe_8e_n50/epoch-2500.pt"
    solver_native_size = 50
    solver_model_params = DEFAULT_MODEL_PARAMS.copy()
    solver_model_params['num_experts'] = 8
    solver_model_params['problem'] = instance_problem_type

    ovrp_passed = False
    if load_dataset and get_env:
        try:
            # Load Data
            logger.info(f"Loading instance {instance_number} from {dataset_path}")
            if not os.path.exists(dataset_path): raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            all_instances = load_dataset(dataset_path, disable_print=True)
            if not all_instances or instance_number >= len(all_instances):
                raise ValueError(f"Instance {instance_number} not found or dataset empty.")
            instance_data_tuple = all_instances[instance_number]

            logger.info(f"Loading solution {instance_number} from {solution_path}")
            if not os.path.exists(solution_path): raise FileNotFoundError(f"Solution file not found: {solution_path}")
            all_solutions = load_dataset(solution_path, disable_print=True)
            if not all_solutions or instance_number >= len(all_solutions):
                raise ValueError(f"Solution {instance_number} not found or solution file empty.")
            solution_entry = all_solutions[instance_number]
            if isinstance(solution_entry, (list, tuple)) and len(solution_entry) >= 1:
                 or_tools_solution_cost = solution_entry[0]
                 or_tools_solution_path = solution_entry[1] if len(solution_entry) > 1 else "N/A"
            elif isinstance(solution_entry, (int, float)):
                 or_tools_solution_cost = solution_entry
                 or_tools_solution_path = "N/A"
            else:
                 raise ValueError(f"Unexpected solution format: {solution_entry}")
            logger.info(f"OR-Tools Solution: Cost={or_tools_solution_cost:.4f}")

            # Load Model (reuse if already loaded)
            if solver_model_instance is None or solver_model_instance.model_params.get('num_experts') != 8:
                logger.info(f"Loading/Re-loading Solver Model: {solver_checkpoint_path}")
                solver_model_instance = load_moe_model(solver_checkpoint_path, device, model_type="MOE", model_params=solver_model_params)
            if not solver_model_instance: raise ValueError("Failed to load the solver model.")
            # Get Env Class
            EnvClassList_test = get_env(instance_problem_type)
            solver_env_class_test = EnvClassList_test[0] if EnvClassList_test else None
            if not solver_env_class_test: raise ValueError(f"Failed env class for {instance_problem_type}")
            # Prepare Data
            instance_batch_list = [instance_data_tuple]
            padded_batch_tuples, target_pad_size = pad_subproblem_batch(instance_batch_list, instance_problem_type)
            padded_batch_tensor_data = prepare_batch_tensor_data(padded_batch_tuples, instance_problem_type, device)
            if not padded_batch_tensor_data: raise ValueError("Failed tensor prep")
            # Solve
            logger.info("Calling solve_vrp_batch...")
            solver_results = solve_vrp_batch(
                solver_model=solver_model_instance,
                solver_env_class=solver_env_class_test,
                original_instance_tuples=instance_batch_list, # <<< Pass the original tuples
                padded_batch_data=padded_batch_tensor_data,
                padded_problem_size=target_pad_size,
                problem_type=instance_problem_type,
                device=device,
                aug_factor=8 # Use standard augmentation
            )

            # Check Result
            if solver_results and solver_results[0][0] != float('inf'):
                model_cost = solver_results[0][0]
                gap = ((model_cost - or_tools_solution_cost) / or_tools_solution_cost) * 100 if or_tools_solution_cost > 1e-6 else float('inf')
                print(f"  Result: PASS - Model Cost={model_cost:.4f} (OR-Tools: {or_tools_solution_cost:.4f}, Gap: {gap:.2f}%)")
                ovrp_passed = True
                test_passed_count += 1
            else:
                print(f"  Result: FAIL - Solver failed or returned no results. Cost: {solver_results[0][0] if solver_results else 'N/A'}")
        except Exception as e:
            print(f"  Result: FAIL - Error during test: {e}")
            logger.exception("OVRPBLTW test error details:", exc_info=True) # Log full traceback
    else:
        print("SKIPPING OVRPBLTW Test: load_dataset or get_env not available.")
    if not ovrp_passed: test_failed_count += 1

    # --- Test 5.2: CVRP ---
    print("Test 5.2: CVRP instance 0")
    instance_problem_type_cvrp = "CVRP"
    dataset_path_cvrp = os.path.join(dataset_base, 'CVRP', 'cvrp50_uniform.pkl')
    instance_number_cvrp = 0
    solution_path_cvrp = os.path.join(dataset_base, 'CVRP', 'hgs_cvrp50_uniform.pkl') # HGS solution for CVRP50
    # Reuse the 8e solver or load a different one if preferred
    # solver_checkpoint_path_cvrp = r"pretrained/mvmoe_4e_n50/epoch-5000.pt"
    # solver_model_params_cvrp = DEFAULT_MODEL_PARAMS.copy(); solver_model_params_cvrp['num_experts'] = 4
    # solver_model_params_cvrp['problem'] = instance_problem_type_cvrp
    solver_checkpoint_path_cvrp = solver_checkpoint_path # Reuse 8e model
    solver_model_params_cvrp = solver_model_params # Reuse 8e params
    solver_native_size_cvrp = 50

    cvrp_passed = False
    if load_dataset and get_env:
        try:
            # Load Data
            logger.info(f"Loading instance {instance_number_cvrp} from {dataset_path_cvrp}")
            if not os.path.exists(dataset_path_cvrp): raise FileNotFoundError(f"Dataset file not found: {dataset_path_cvrp}")
            all_instances_cvrp = load_dataset(dataset_path_cvrp, disable_print=True)
            if not all_instances_cvrp or instance_number_cvrp >= len(all_instances_cvrp):
                raise ValueError(f"Instance {instance_number_cvrp} not found or dataset empty.")
            instance_data_tuple_cvrp = all_instances_cvrp[instance_number_cvrp]

            logger.info(f"Loading solution {instance_number_cvrp} from {solution_path_cvrp}")
            if not os.path.exists(solution_path_cvrp): raise FileNotFoundError(f"Solution file not found: {solution_path_cvrp}")
            all_solutions_cvrp = load_dataset(solution_path_cvrp, disable_print=True)
            if not all_solutions_cvrp or instance_number_cvrp >= len(all_solutions_cvrp):
                raise ValueError(f"Solution {instance_number_cvrp} not found or solution file empty.")
            solution_entry_cvrp = all_solutions_cvrp[instance_number_cvrp]
            if isinstance(solution_entry_cvrp, (list, tuple)): or_tools_solution_cost_cvrp = solution_entry_cvrp[0]
            else: or_tools_solution_cost_cvrp = solution_entry_cvrp
            # Load Model (reuse if same checkpoint/params)
            if solver_model_instance is None or solver_checkpoint_path_cvrp != solver_checkpoint_path:
                logger.info(f"Loading/Re-loading Solver Model: {solver_checkpoint_path_cvrp}")
                solver_model_instance = load_moe_model(solver_checkpoint_path_cvrp, device, model_type="MOE", model_params=solver_model_params_cvrp)
            if not solver_model_instance: raise ValueError("Failed to load solver model for CVRP")
            # Get Env Class
            EnvClassList_cvrp = get_env(instance_problem_type_cvrp)
            solver_env_class_cvrp = EnvClassList_cvrp[0] if EnvClassList_cvrp else None
            if not solver_env_class_cvrp: raise ValueError(f"Failed env class for {instance_problem_type_cvrp}")
            # Prepare Data
            instance_batch_list_cvrp = [instance_data_tuple_cvrp]
            padded_batch_tuples_cvrp, target_pad_size_cvrp = pad_subproblem_batch(instance_batch_list_cvrp, instance_problem_type_cvrp)
            padded_batch_tensor_data_cvrp = prepare_batch_tensor_data(padded_batch_tuples_cvrp, instance_problem_type_cvrp, device)
            if not padded_batch_tensor_data_cvrp: raise ValueError("Failed tensor prep for CVRP")
            # Solve
            logger.info("Calling solve_vrp_batch...")
            solver_results_cvrp = solve_vrp_batch(
                solver_model=solver_model_instance,
                solver_env_class=solver_env_class_cvrp,
                original_instance_tuples=instance_batch_list_cvrp,
                padded_batch_data=padded_batch_tensor_data_cvrp,
                padded_problem_size=target_pad_size_cvrp,
                problem_type=instance_problem_type_cvrp,
                device=device,
                aug_factor=8
            )

            # Check Result
            if solver_results_cvrp and solver_results_cvrp[0][0] != float('inf'):
                model_cost_cvrp = solver_results_cvrp[0][0]
                gap_cvrp = ((model_cost_cvrp - or_tools_solution_cost_cvrp) / or_tools_solution_cost_cvrp) * 100 if or_tools_solution_cost_cvrp > 1e-6 else float('inf')
                print(f"  Result: PASS - Model Cost={model_cost_cvrp:.4f} (HGS: {or_tools_solution_cost_cvrp:.4f}, Gap: {gap_cvrp:.2f}%)")
                cvrp_passed = True
                test_passed_count += 1
            else:
                print(f"  Result: FAIL - Solver failed or returned no results. Cost: {solver_results_cvrp[0][0] if solver_results_cvrp else 'N/A'}")
        except Exception as e:
            print(f"  Result: FAIL - Error during CVRP test: {e}")
            logger.exception("CVRP test error details:", exc_info=True)
    else:
        print("SKIPPING CVRP Test: load_dataset or get_env not available.")
    if not cvrp_passed: test_failed_count += 1

    print(f"solve_vrp_batch Tests: {test_passed_count} Passed, {test_failed_count} Failed")
    total_passed += test_passed_count
    total_failed += test_failed_count
    print("=" * 60)

    # --- Test partition_instance ---
    print("\n--- Testing partition_instance ---")
    test_passed_count = 0 # Reset for this test section
    test_failed_count = 0
    partition_instance_passed = False # Track pass/fail for this specific test

    # Setup parameters based on user request
    dataset_base = './data' # Assuming 'data' is in the current working dir or adjust path
    partition_dataset_path_cvrp = os.path.join(dataset_base, 'CVRP', 'cvrp500_uniform.pkl')
    instance_number_cvrp = 0
    # IMPORTANT: Construct the correct relative path if needed
    partitioner_model_path = os.path.join('results','TAM_MoE','MOE_4e_CVRP_n200_20250425_050856','40.pt')
    test_merge_num = 3
    problem_type = "CVRP"
    # --- Define Partitioner Model Parameters ---
    # Based on the path 'MOE_4e_CVRP_n200...', assuming it's an MOEModel
    test_partitioner_params = {
        "model_type": "MOE", # Use the correct solver model type
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128**(1/2),
        "encoder_layer_num": 6,
        "decoder_layer_num": 1,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10.0,
        "ff_hidden_dim": 512,
        "num_experts": 4, # From '4e' in path
        "eval_type": "argmax", # Use argmax for deterministic partitioning during inference
        "norm": "instance",
        "norm_loc": "norm_last",
        "topk": 2, # Common default
        "expert_loc": ['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'], # Common default
        "routing_level": "node", # Common default
        "routing_method": "input_choice", # Common default
        "problem": problem_type, # Use the specific problem type for the instance
    }
    print(f"Using device: {device}")
    print(f"Loading instance {instance_number_cvrp} from: {partition_dataset_path_cvrp}")
    print(f"Loading fine-tuned model from: {partitioner_model_path}")
    print(f"Merge num: {test_merge_num}")
    print(f"Using model parameters: {test_partitioner_params}")

    try:
        full_dataset_cvrp = load_dataset(partition_dataset_path_cvrp)
        if not full_dataset_cvrp or instance_number_cvrp >= len(full_dataset_cvrp):
            print(f"Error: Could not load dataset or instance number {instance_number_cvrp} is out of bounds.")
            test_failed_count += 1
        else:
            test_instance_cvrp = full_dataset_cvrp[instance_number_cvrp]

            # Call the function
            subproblem_tuples, raw_seq = partition_instance(
                original_instance_tuple=test_instance_cvrp,
                problem_type=problem_type,
                partitioner_checkpoint_path=partitioner_model_path,
                merge_num=test_merge_num,
                device=device,
                partitioner_model_params=test_partitioner_params # Pass solver params
            )

            # Print results
            if subproblem_tuples is not None:
                print(f"\n  Result: PASS - Partitioning successful!")
                print(f"    Raw sequence generated (first 50): {raw_seq[:50] if raw_seq else 'N/A'}")
                print(f"    Number of subproblems created: {len(subproblem_tuples)}")
                if subproblem_tuples:
                    print("    Details of first few subproblems:")
                    for i, sub_tuple in enumerate(subproblem_tuples[:5]):
                         # Assuming format: (depot_xy, node_xy, demand, capacity, ...)
                         num_nodes_in_sub = len(sub_tuple[1]) if len(sub_tuple)>1 and sub_tuple[1] is not None else 0
                         print(f"      Subproblem {i+1}: {num_nodes_in_sub} customer nodes")
                test_passed_count += 1
                partition_instance_passed = True
            else:
                print(f"\n  Result: FAIL - Partitioning failed.")
                print(f"    Raw sequence generated (if any): {raw_seq}")
                test_failed_count += 1


    except FileNotFoundError:
        print(f"Error: Dataset or Model file not found. Please check paths:")
        print(f"  Dataset: {partition_dataset_path_cvrp}")
        print(f"  Model: {partitioner_model_path}")
        test_failed_count += 1
    except ImportError:
        print(f"Error: Could not import necessary modules (e.g., MOEModel). Check imports.")
        test_failed_count += 1
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
        test_failed_count += 1

    print(f"\npartition_instance Test: {test_passed_count} Passed, {test_failed_count} Failed")
    total_passed += test_passed_count
    total_failed += test_failed_count
    print("\n--- Finished testing partition_instance ---")

    # Grand Total
    print(f"\nOverall Test Summary: {total_passed} Passed, {total_failed} Failed")
    print("="*60)