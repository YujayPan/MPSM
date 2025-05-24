import os
import sys
import torch
import pickle
import json
import time
import matplotlib.pyplot as plt
import numpy as np # Added for normalization
import math # Added for cost calculation

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cvrp_lib_integration.cvrp_lib_parser import parse_vrp_file, to_internal_tuple
from utils import VRP_DATA_FORMAT, seed_everything, get_env
from partitioner_solver_utils import (
    load_moe_model, DEFAULT_MODEL_PARAMS,
    _split_sequence_by_zeros, # For initial split
    merge_subproblems_by_centroid_fixed_size, # For merging
    create_subproblem_instance, # For creating subproblem tuples for solver
    pad_subproblem_batch, prepare_batch_tensor_data, solve_vrp_batch # For NN solver
)
# from data_visualize import visualize_vrp_instance, visualize_solution # For visualization
# from ortools_solver import ortools_solve_vrp # If using OR-Tools

# --- Configuration --- 
SEED = 2024
# CVRP_LIB_FILE_PATH = os.path.join(project_root, "CVRP-LIB", "P-n16-k8.vrp") # Smaller instance for faster testing
CVRP_LIB_FILE_PATH = os.path.join(project_root, "CVRP-LIB", "X-n101-k25.vrp")
DEVICE_STR = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_STR)

# --- Partitioner Model Config (Fine-tuned Solver for Partitioning) ---
# !!! USER: Update this path to your actual partitioner model checkpoint !!!
PARTITIONER_CHECKPOINT_PATH = "pretrained/mvmoe_4e_n100/epoch-5000.pt" # Example placeholder
PARTITIONER_MODEL_TYPE = "MOE" # Example
PARTITIONER_NUM_EXPERTS = 4    # Example
# Add other necessary partitioner model params from generate_partitions.py args
PARTITIONER_EMBEDDING_DIM = 128
PARTITIONER_ENCODER_LAYER_NUM = 6
PARTITIONER_FF_HIDDEN_DIM = 512

# --- Main Solver Model Config ---
# !!! USER: Update this path to your actual main solver model checkpoint !!!
SOLVER_CHECKPOINT_PATH = "pretrained/mvmoe_8e_n50/epoch-2500.pt" # Example placeholder
SOLVER_MODEL_TYPE = "MOE" # Example
SOLVER_NUM_EXPERTS = 8    # Example
# Add other necessary solver model params from solve_from_partitions.py args
SOLVER_EMBEDDING_DIM = 128
SOLVER_ENCODER_LAYER_NUM = 6
SOLVER_FF_HIDDEN_DIM = 512
SOLVER_AUG_FACTOR = 1 # For faster testing, can increase later

# --- Merge Config ---
MERGE_CONFIG_NAME = 'adaptive' # Options: 'raw_subroutes', 'm1', 'm3', 'adaptive', 'adaptive_sX' (e.g., 'adaptive_s50')

# --- Output Directory (Optional, for saving intermediate files) ---
OUTPUT_DIR = os.path.join(project_root, "cvrp_lib_integration", "run_outputs_single_instance")
RUN_ID = f"single_test_{time.strftime('%Y%m%d_%H%M%S')}"
INSTANCE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, RUN_ID)

#----------------------------------------------------------------------
# Helper function extracted and adapted from generate_partitions.py
#----------------------------------------------------------------------
def generate_sequence_and_initial_routes_from_tuple(
    original_instance_tuple, 
    problem_type, 
    loaded_partitioner_model, # Expects pre-loaded model
    device_obj, # Expects torch.device object
    max_seq_len_factor=2
):
    """
    Generates a node sequence using the partitioner model and splits it by zeros.
    Adapted from generate_partitions.py to work with a single pre-loaded tuple.
    """
    raw_sequence = None
    initial_subproblem_node_lists = []

    try:
        node_xy_index = VRP_DATA_FORMAT[problem_type].index('node_xy')
        num_customer_nodes = len(original_instance_tuple[node_xy_index])
        if num_customer_nodes <= 0:
            print("Error: Instance has no customer nodes.")
            return None, []
        
        max_seq_len = max_seq_len_factor * (num_customer_nodes + 1)

        EnvClassList = get_env(problem_type)
        if not EnvClassList:
            print(f"Error: Could not get env class for {problem_type}")
            return None, []
        PartitionEnvClass = EnvClassList[0]
        
        env_params = {"problem_size": num_customer_nodes, "pomo_size": 1, "device": device_obj}
        partition_env = PartitionEnvClass(**env_params)
        
        padded_batch_tuples, target_pad_size = pad_subproblem_batch(
            [original_instance_tuple], problem_type, num_customer_nodes
        )
        if not padded_batch_tuples or target_pad_size != num_customer_nodes:
            print(f"Error: Padding/Target size mismatch. Expected {num_customer_nodes}, got {target_pad_size}")
            return None, []
        
        instance_tensor_data = prepare_batch_tensor_data(padded_batch_tuples, problem_type, device_obj)
        if not instance_tensor_data:
            print("Error: Failed to prepare instance tensor data for partitioning.")
            return None, []
        
        partition_env.load_problems(batch_size=1, problems=instance_tensor_data, aug_factor=1)

        loaded_partitioner_model.eval()
        loaded_partitioner_model.set_eval_type('argmax')

        with torch.no_grad():
            reset_state, _, _ = partition_env.reset()
            loaded_partitioner_model.pre_forward(reset_state)
            state, _, done = partition_env.pre_step()
            step_count = 0
            while not done and step_count < max_seq_len:
                selected, _ = loaded_partitioner_model(state)
                state, _, done = partition_env.step(selected)
                step_count += 1
            
            if step_count >= max_seq_len:
                print(f"Warning: Sequence generation reached max length ({max_seq_len}).")
            
            if hasattr(partition_env, 'selected_node_list') and partition_env.selected_node_list is not None:
                if partition_env.selected_node_list.numel() > 0:
                    raw_sequence = partition_env.selected_node_list.view(-1).cpu().tolist()
                else:
                    print("Warning: partition_env.selected_node_list is empty.")
            else:
                print("Warning: partition_env.selected_node_list not found.")

        if raw_sequence is None:
            print("Error: Failed to generate raw sequence from environment rollout.")
            return None, []
        
        initial_subproblem_node_lists = _split_sequence_by_zeros(raw_sequence)

    except Exception as e:
        print(f"Error in generate_sequence_and_initial_routes_from_tuple: {e}")
        import traceback
        traceback.print_exc()
        return None, []
            
    return raw_sequence, initial_subproblem_node_lists

#----------------------------------------------------------------------
# Helper function extracted and adapted from solve_from_partitions.py
#----------------------------------------------------------------------
def run_partition_solver_for_single_instance(
    original_instance_tuple_for_processing, # This tuple might be normalized
    problem_type, 
    # subproblem_node_lists_from_merge: List of lists, where each inner list 
    # contains 1-based original node indices for that subproblem.
    # This is crucial for remapping paths later.
    subproblem_node_lists_from_merge, 
    solver_model, 
    device, 
    solver_aug_factor,
    solver_model_params 
):
    """
    Solves an original instance by creating subproblem instances from pre-loaded 
    node lists, then solving these subproblems using the NN Solver.
    The original_instance_tuple_for_processing is used to create subproblems and should have the scale
    expected by the model (e.g., normalized if the model was trained on normalized data).
    Returns a dict including score (normalized), time, num_subproblems, and full_path_original_indices.
    """
    if not original_instance_tuple_for_processing or not solver_model or not subproblem_node_lists_from_merge:
        print("Error run_partition_solver: Missing original instance, model, or subproblem_node_lists_from_merge.")
        return {'score': float('inf'), 'solve_time_seconds': 0, 'num_subproblems': 0, 'error': 'MissingInput', 'full_path_original_indices': []}

    solve_time_start = time.time()
    
    EnvClassList = get_env(problem_type)
    if not EnvClassList:
        print(f"Error: Could not get env class for solver, problem type {problem_type}")
        return {'score': float('inf'), 'solve_time_seconds': 0, 'num_subproblems': 0, 'error': 'NoEnvClass', 'full_path_original_indices': []}
    SolverEnvClass = EnvClassList[0]

    subproblem_instance_tuples = []
    num_original_nodes_in_processed_instance = 0
    try:
        node_xy_idx_orig = VRP_DATA_FORMAT[problem_type].index('node_xy')
        num_original_nodes_in_processed_instance = len(original_instance_tuple_for_processing[node_xy_idx_orig]) 
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error run_partition_solver: Error accessing node_xy for count: {e}")
        return {'score': float('inf'), 'solve_time_seconds': 0, 'num_subproblems': 0, 'error': 'OriginalNodeAccessError', 'full_path_original_indices': []}

    # subproblem_node_lists_from_merge contains the original 1-based indices for each subproblem
    for original_indices_for_current_sub in subproblem_node_lists_from_merge:
        if not original_indices_for_current_sub: continue
        
        # Validate that these original indices are within bounds of the *original* problem,
        # not necessarily the num_original_nodes_in_processed_instance if processing involved truncation (not typical here).
        # For CVRP-LIB, num_original_nodes_in_processed_instance should be the actual number of customers.
        valid_indices_for_sub = [idx for idx in original_indices_for_current_sub if 1 <= idx <= num_original_nodes_in_processed_instance]
        if not valid_indices_for_sub: continue
        
        sub_instance = create_subproblem_instance(original_instance_tuple_for_processing, problem_type, valid_indices_for_sub)
        if sub_instance:
            subproblem_instance_tuples.append(sub_instance)
        else:
            print(f"Warning: Failed to create subproblem instance for node list: {valid_indices_for_sub}")

    if not subproblem_instance_tuples:
        print(f"Warning: No valid subproblem instances created from node lists for {problem_type}.")
        return {'score': float('inf'), 'solve_time_seconds': time.time() - solve_time_start, 'num_subproblems': 0, 'error': 'NoValidSubproblemsCreated', 'full_path_original_indices': []}
    
    num_actual_subproblems = len(subproblem_instance_tuples)

    max_nodes_in_sub_batch = 0
    node_xy_idx_sub = VRP_DATA_FORMAT[problem_type].index('node_xy')
    for sub_inst_tuple in subproblem_instance_tuples:
        max_nodes_in_sub_batch = max(max_nodes_in_sub_batch, len(sub_inst_tuple[node_xy_idx_sub]))
    
    target_pad_size_for_subs = max(1, max_nodes_in_sub_batch)
    
    padded_subproblem_batch, actual_pad_size_for_subs = pad_subproblem_batch(
        subproblem_instance_tuples, problem_type, target_pad_size_for_subs
    )
    if not padded_subproblem_batch or len(padded_subproblem_batch) != num_actual_subproblems:
        print("Error: Failed to pad subproblem batch correctly.")
        return {'score': float('inf'), 'solve_time_seconds': time.time() - solve_time_start, 'num_subproblems': num_actual_subproblems, 'error': 'SubproblemPaddingError', 'full_path_original_indices': []}

    padded_subproblem_tensor_data = prepare_batch_tensor_data(
        padded_subproblem_batch, problem_type, device
    )
    if not padded_subproblem_tensor_data:
        print("Error: Failed to prepare tensor data for subproblem batch.")
        return {'score': float('inf'), 'solve_time_seconds': time.time() - solve_time_start, 'num_subproblems': num_actual_subproblems, 'error': 'SubproblemTensorPrepError', 'full_path_original_indices': []}

    solver_model.eval()
    flat_solver_results = solve_vrp_batch( # List of (sub_score, sub_path_nodes_in_subproblem)
        solver_model=solver_model,
        solver_env_class=SolverEnvClass,
        original_instance_tuples=subproblem_instance_tuples, 
        padded_batch_data=padded_subproblem_tensor_data,
        padded_problem_size=actual_pad_size_for_subs,
        problem_type=problem_type,
        device=device,
        aug_factor=solver_aug_factor
    )

    if not flat_solver_results or len(flat_solver_results) != num_actual_subproblems:
        print(f"Error: Subproblem solver results count mismatch. Expected {num_actual_subproblems}, got {len(flat_solver_results) if flat_solver_results else 0}.")
        error_results_fill = [(float('inf'), [])] * (num_actual_subproblems - (len(flat_solver_results) if flat_solver_results else 0)) # path is empty list
        flat_solver_results = (flat_solver_results if flat_solver_results else []) + error_results_fill

    total_aggregated_score_normalized = 0
    full_reconstructed_path_original_indices = []
    path_reconstruction_failed = False

    for i, (sub_score, sub_path_nodes_in_subproblem) in enumerate(flat_solver_results):
        if sub_score == float('inf') or sub_score is None or torch.isnan(torch.tensor(sub_score)):
            total_aggregated_score_normalized = float('inf')
            path_reconstruction_failed = True # If one subproblem fails, path is incomplete
            print(f"Warning: Subproblem {i} failed to solve (score: {sub_score}). Instance score is inf.")
            break # Stop aggregating score and path
        total_aggregated_score_normalized += sub_score

        # Reconstruct path for this subproblem using original node indices
        if sub_path_nodes_in_subproblem is None: # Should be caught by score check, but defensive
            path_reconstruction_failed = True
            print(f"Warning: Subproblem {i} has valid score but no path. Path reconstruction failed.")
            break
        
        # original_indices_for_current_sub are the 1-based original node IDs for the i-th subproblem
        original_indices_for_current_sub = subproblem_node_lists_from_merge[i]
        remapped_segment = []
        for node_idx_in_sub in sub_path_nodes_in_subproblem:
            if node_idx_in_sub == 0: # Depot
                remapped_segment.append(0)
            else: # Customer node (1-based within this subproblem)
                if 1 <= node_idx_in_sub <= len(original_indices_for_current_sub):
                    original_node_idx = original_indices_for_current_sub[node_idx_in_sub - 1]
                    remapped_segment.append(original_node_idx)
                else:
                    print(f"Warning: Invalid node index {node_idx_in_sub} in sub_path for subproblem {i}. Max original nodes in this sub: {len(original_indices_for_current_sub)}. Path: {sub_path_nodes_in_subproblem}")
                    path_reconstruction_failed = True
                    break 
        if path_reconstruction_failed: break

        # Concatenate remapped_segment to full_reconstructed_path_original_indices
        if not full_reconstructed_path_original_indices:
            full_reconstructed_path_original_indices.extend(remapped_segment)
        elif remapped_segment: # Only extend if there's something to add
            if full_reconstructed_path_original_indices[-1] == 0 and remapped_segment[0] == 0:
                full_reconstructed_path_original_indices.extend(remapped_segment[1:])
            elif full_reconstructed_path_original_indices[-1] != 0 and remapped_segment[0] != 0:
                print(f"Warning: Concatenating subproblem paths that don't adhere to depot-ending/starting. Sub-path: {remapped_segment}. Full path ends with: {full_reconstructed_path_original_indices[-1]}. Inserting depot.")
                full_reconstructed_path_original_indices.append(0)
                full_reconstructed_path_original_indices.extend(remapped_segment)
            else:
                full_reconstructed_path_original_indices.extend(remapped_segment)
    
    if path_reconstruction_failed:
        full_reconstructed_path_original_indices = [] # Clear path if reconstruction failed

    current_solve_time = time.time() - solve_time_start

    return {
        'score': total_aggregated_score_normalized, 
        'solve_time_seconds': current_solve_time,
        'num_subproblems': num_actual_subproblems,
        'full_path_original_indices': full_reconstructed_path_original_indices # List of 1-based original indices
    }

def calculate_path_cost_original_coords(flat_path_original_indices, depot_xy_raw_list, nodes_xy_raw_list):
    """
    Calculates the total travel distance for a given flat path using original (unnormalized) coordinates.
    Args:
        flat_path_original_indices (list[int]): A single list representing the sequence of visits.
                                                 0 for depot, 1-N for original customer indices.
                                                 Example: [0, 5, 12, 0, 3, 28, 0]
        depot_xy_raw_list (list[list[float]]): Original depot coordinates, e.g., [[x, y]]
        nodes_xy_raw_list (list[list[float]]): List of original customer coordinates, e.g., [[x1,y1], ..., [xN,yN]]
    Returns:
        float: The total Euclidean distance of the path.
    """
    if not flat_path_original_indices:
        return 0.0

    all_coords_raw = [depot_xy_raw_list[0]] + nodes_xy_raw_list # Depot is at index 0, customer k (1-based) is at index k
    
    total_distance = 0.0
    
    if len(flat_path_original_indices) < 2:
        return 0.0

    for i in range(len(flat_path_original_indices) - 1):
        from_node_original_idx = flat_path_original_indices[i]
        to_node_original_idx = flat_path_original_indices[i+1]
        
        try:
            coord_from = all_coords_raw[from_node_original_idx] 
            coord_to = all_coords_raw[to_node_original_idx]
        except IndexError:
            print(f"Error: Node index out of bounds during cost calculation. From: {from_node_original_idx}, To: {to_node_original_idx}. Max index: {len(all_coords_raw)-1}")
            return float('inf')

        dist = math.sqrt((coord_from[0] - coord_to[0])**2 + (coord_from[1] - coord_to[1])**2)
        total_distance += dist
        
    return total_distance

def main():
    seed_everything(SEED)
    os.makedirs(INSTANCE_OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE_STR}")
    print(f"Processing single instance: {CVRP_LIB_FILE_PATH}")
    print(f"Output will be in: {INSTANCE_OUTPUT_DIR}")

    # 1. Load and Parse CVRP-LIB Instance
    parsed_data = parse_vrp_file(CVRP_LIB_FILE_PATH)
    if not parsed_data:
        print(f"Failed to parse {CVRP_LIB_FILE_PATH}. Exiting.")
        return
    
    # Store original unnormalized data for final cost calculation
    original_instance_tuple_RAW = to_internal_tuple(parsed_data)
    if not original_instance_tuple_RAW:
        print("Failed to convert to internal tuple (RAW). Exiting.")
        return
    
    problem_type = parsed_data.get("problem_type", "CVRP")
    instance_name = parsed_data.get("name", "unknown_instance")
    num_customer_nodes_original = len(original_instance_tuple_RAW[1]) # Based on raw data
    print(f"Instance: {instance_name}, Type: {problem_type}, Customers: {num_customer_nodes_original}")

    with open(os.path.join(INSTANCE_OUTPUT_DIR, f"{instance_name}_original_RAW.pkl"), 'wb') as f:
        pickle.dump(original_instance_tuple_RAW, f)

    # --- Coordinate Normalization --- 
    print("\nNormalizing coordinates to [0,1] range for model processing...")
    depot_coords_raw_for_norm = np.array(original_instance_tuple_RAW[0]) 
    node_coords_raw_for_norm = np.array(original_instance_tuple_RAW[1])  
    
    if node_coords_raw_for_norm.size == 0:
        print("Warning: No customer nodes to normalize.")
        instance_for_processing = original_instance_tuple_RAW # Use raw if no customers
    else:
        all_coords_raw_for_norm = np.vstack((depot_coords_raw_for_norm, node_coords_raw_for_norm))
        min_coords = all_coords_raw_for_norm.min(axis=0)
        max_coords = all_coords_raw_for_norm.max(axis=0)
        range_coords = max_coords - min_coords
        range_coords[range_coords == 0] = 1.0 
        
        normalized_depot_list = ((depot_coords_raw_for_norm - min_coords) / range_coords).tolist()
        normalized_node_list = ((node_coords_raw_for_norm - min_coords) / range_coords).tolist()
        
        # Create the instance tuple that will be fed to the models
        instance_for_processing = (
            normalized_depot_list, 
            normalized_node_list,  
            original_instance_tuple_RAW[2], # demand (remains unnormalized relative to capacity for prepare_batch_tensor_data)
            original_instance_tuple_RAW[3]  # capacity
        )
        # Add other fields if present in original_instance_tuple_RAW (e.g. for VRPTW)
        if len(original_instance_tuple_RAW) > 4:
            instance_for_processing += original_instance_tuple_RAW[4:]

        print(f"Normalization complete. Min Coords: {min_coords}, Range: {range_coords}")
        with open(os.path.join(INSTANCE_OUTPUT_DIR, f"{instance_name}_normalized_input.pkl"), 'wb') as f:
            pickle.dump(instance_for_processing, f)

    # 2. Load Partitioner Model
    print(f"\n--- Loading Partitioner Model ({PARTITIONER_MODEL_TYPE}) ---")
    print(f"Checkpoint: {PARTITIONER_CHECKPOINT_PATH}")
    partitioner_model_params_dict = DEFAULT_MODEL_PARAMS.copy()
    partitioner_model_params_dict['model_type'] = PARTITIONER_MODEL_TYPE
    partitioner_model_params_dict['num_experts'] = PARTITIONER_NUM_EXPERTS
    partitioner_model_params_dict['device'] = DEVICE
    partitioner_model_params_dict['embedding_dim'] = PARTITIONER_EMBEDDING_DIM
    partitioner_model_params_dict['ff_hidden_dim'] = PARTITIONER_FF_HIDDEN_DIM
    partitioner_model_params_dict['encoder_layer_num'] = PARTITIONER_ENCODER_LAYER_NUM
    
    partitioner_model = load_moe_model(
        PARTITIONER_CHECKPOINT_PATH, 
        DEVICE, 
        model_type=PARTITIONER_MODEL_TYPE,
        model_params=partitioner_model_params_dict
    )
    if not partitioner_model:
        print("Failed to load partitioner model. Exiting.")
        return
    print("Partitioner model loaded.")

    # 3. Generate Initial Partitions
    print("\n--- Generating Initial Partitions (using processed coordinates) ---")
    partition_time_start = time.time()
    raw_sequence, initial_subproblem_node_lists = generate_sequence_and_initial_routes_from_tuple(
        instance_for_processing, 
        problem_type,
        partitioner_model,
        DEVICE
    )
    partition_time_seconds = time.time() - partition_time_start

    if raw_sequence is None or initial_subproblem_node_lists is None:
        print("Failed to generate raw sequence/initial_subproblem_node_lists. Exiting.")
        return
    
    print(f"Raw sequence generated (len {len(raw_sequence)}): {raw_sequence[:30]}...")
    print(f"Initial subproblems created: {len(initial_subproblem_node_lists)} (list of 1-based original node lists)")
    print(f"Partitioning time: {partition_time_seconds:.4f}s")
    with open(os.path.join(INSTANCE_OUTPUT_DIR, f"{instance_name}_raw_sequence.pkl"), 'wb') as f:
        pickle.dump(raw_sequence, f)
    with open(os.path.join(INSTANCE_OUTPUT_DIR, f"{instance_name}_initial_subproblems_orig_indices.pkl"), 'wb') as f:
        pickle.dump(initial_subproblem_node_lists, f) # These are lists of original 1-based indices

    # 4. Merge Subproblems
    print(f"\n--- Merging Subproblems (Config: {MERGE_CONFIG_NAME}, using processed coordinates for centroids) ---")
    def parse_merge_config_string(config_name: str):
        if config_name == 'raw_subroutes': return None, None 
        elif config_name == 'm1': return 1, 0
        elif config_name == 'm3': return 3, 0
        elif config_name == 'adaptive': return -1, 0
        elif config_name.startswith('adaptive_s'):
            try: return -1, int(config_name.split('adaptive_s')[-1])
            except ValueError: return -1, 0
        else: return None, None

    parsed_merge_num, parsed_adaptive_target = parse_merge_config_string(MERGE_CONFIG_NAME)
    
    merged_node_lists_for_solve = [] # This will be a list of lists of original 1-based indices
    if parsed_merge_num is None and MERGE_CONFIG_NAME != 'raw_subroutes':
        print(f"Unknown merge config: {MERGE_CONFIG_NAME}. Skipping merge, using initial subproblems.")
        merged_node_lists_for_solve = initial_subproblem_node_lists
    elif MERGE_CONFIG_NAME == 'raw_subroutes':
        print("Using raw subroutes directly for solving (no merge applied).")
        merged_node_lists_for_solve = initial_subproblem_node_lists
    else:
        merge_time_start = time.time()
        # merge_subproblems_by_centroid_fixed_size expects original_loc and original_depot to be from the
        # same scale as the initial_subproblems's indices refer to.
        # Here, initial_subproblem_node_lists contains ORIGINAL 1-based indices.
        # However, centroids should be calculated on a consistent scale, ideally normalized if partitioning was.
        # The `merge_subproblems_by_centroid_fixed_size` itself takes `original_loc` which could be normalized.
        # The key is that `initial_subproblems` (node IDs) and `original_loc` (coords for those IDs) must match.
        # Since `initial_subproblem_node_lists` are original IDs, we should use normalized coords for centroid calculations
        # if the partitioning model operated on normalized data.
        
        # Use normalized coordinates for centroid calculation in merge
        # The `initial_subproblem_node_lists` still contains original 1-based indices.
        # `merge_subproblems_by_centroid_fixed_size` will use these indices to look up coordinates
        # in `instance_for_processing[1]` (normalized node_xy) for centroid calculation.
        merged_node_lists_for_solve = merge_subproblems_by_centroid_fixed_size(
            initial_subproblems=initial_subproblem_node_lists, # List of lists of ORIGINAL 1-based indices
            original_loc=instance_for_processing[1],      # NORMALIZED node_xy for centroid calculation
            original_depot=instance_for_processing[0][0], # NORMALIZED depot_xy[0] for centroid calculation
            problem_size_for_dynamic_target=num_customer_nodes_original, 
            merge_num=parsed_merge_num,
            target_node_count=parsed_adaptive_target,
            problem_type=problem_type
        )
        merge_time_seconds = time.time() - merge_time_start
        print(f"Merged into {len(merged_node_lists_for_solve)} subproblems (list of 1-based original node lists).")
        print(f"Merging time: {merge_time_seconds:.4f}s")
        with open(os.path.join(INSTANCE_OUTPUT_DIR, f"{instance_name}_merged_subproblems_{MERGE_CONFIG_NAME}_orig_indices.pkl"), 'wb') as f:
            pickle.dump(merged_node_lists_for_solve, f) 
    
    if not merged_node_lists_for_solve:
        print("No subproblem lists available for solving. Exiting.")
        return

    # 5. Load Solver Model
    print(f"\n--- Loading Solver Model ({SOLVER_MODEL_TYPE}) ---")
    print(f"Checkpoint: {SOLVER_CHECKPOINT_PATH}")
    solver_model_params_dict = DEFAULT_MODEL_PARAMS.copy()
    solver_model_params_dict['model_type'] = SOLVER_MODEL_TYPE
    solver_model_params_dict['num_experts'] = SOLVER_NUM_EXPERTS
    solver_model_params_dict['device'] = DEVICE
    solver_model_params_dict['embedding_dim'] = SOLVER_EMBEDDING_DIM
    solver_model_params_dict['ff_hidden_dim'] = SOLVER_FF_HIDDEN_DIM
    solver_model_params_dict['encoder_layer_num'] = SOLVER_ENCODER_LAYER_NUM

    solver_model = load_moe_model(
        SOLVER_CHECKPOINT_PATH, 
        DEVICE, 
        model_type=SOLVER_MODEL_TYPE,
        model_params=solver_model_params_dict
    )
    if not solver_model:
        print("Failed to load solver model. Exiting.")
        return
    print("Solver model loaded.")

    # 6. Solve Merged Subproblems using NN Solver
    print(f"\n--- Solving Partitions ({MERGE_CONFIG_NAME}) with NN Solver (using processed coordinates) ---")
    # `merged_node_lists_for_solve` contains lists of original 1-based indices.
    # `run_partition_solver_for_single_instance` will use these to create subproblems from `instance_for_processing`.
    nn_solver_results_dict = run_partition_solver_for_single_instance(
        original_instance_tuple_for_processing=instance_for_processing, 
        problem_type=problem_type,
        subproblem_node_lists_from_merge=merged_node_lists_for_solve,
        solver_model=solver_model,
        device=DEVICE,
        solver_aug_factor=SOLVER_AUG_FACTOR,
        solver_model_params=solver_model_params_dict
    )

    # The 'score' in nn_solver_results_dict is based on normalized coordinates.
    normalized_score = nn_solver_results_dict.get('score')
    solve_time_seconds = nn_solver_results_dict.get('solve_time_seconds')
    num_subproblems_solved = nn_solver_results_dict.get('num_subproblems')
    solver_error = nn_solver_results_dict.get('error')
    # This path uses original 1-based indices, 0 for depot.
    reconstructed_path_orig_indices = nn_solver_results_dict.get('full_path_original_indices', [])


    print("NN Solver Results (Score is based on processed/normalized coordinates):")
    print(f"  Normalized Score: {normalized_score}")
    print(f"  Solve Time: {solve_time_seconds:.4f}s")
    print(f"  Num Subproblems Solved: {num_subproblems_solved}")
    if solver_error:
        print(f"  Error: {solver_error}")
    
    # 7. Calculate cost using original coordinates
    cost_original_coords = float('inf')
    if reconstructed_path_orig_indices:
        cost_original_coords = calculate_path_cost_original_coords(
            reconstructed_path_orig_indices,
            original_instance_tuple_RAW[0], # depot_xy_raw
            original_instance_tuple_RAW[1]  # nodes_xy_raw
        )
        print(f"  Reconstructed Path (Original Indices, len {len(reconstructed_path_orig_indices)}): {reconstructed_path_orig_indices[:50]}...")
        print(f"  Calculated Cost (Original Coordinates): {cost_original_coords:.4f}")
    else:
        print("  No valid path reconstructed to calculate original coordinate cost.")

    # Save results
    output_data_to_save = {
        "instance_name": instance_name,
        "problem_type": problem_type,
        "partitioner_model": PARTITIONER_CHECKPOINT_PATH,
        "solver_model": SOLVER_CHECKPOINT_PATH,
        "merge_config": MERGE_CONFIG_NAME,
        "normalized_score": normalized_score,
        "cost_original_coordinates": cost_original_coords,
        "solve_time_seconds": solve_time_seconds,
        "num_subproblems": num_subproblems_solved,
        "solver_error": solver_error,
        "reconstructed_path_preview": reconstructed_path_orig_indices[:100] # Save a preview
    }
    output_filename = f"{instance_name}_results_{MERGE_CONFIG_NAME}.json"
    with open(os.path.join(INSTANCE_OUTPUT_DIR, output_filename), 'w') as f:
        json.dump(output_data_to_save, f, indent=4)
    
    # Save full path separately if needed
    with open(os.path.join(INSTANCE_OUTPUT_DIR, f"{instance_name}_reconstructed_path_{MERGE_CONFIG_NAME}.pkl"), 'wb') as f:
        pickle.dump(reconstructed_path_orig_indices, f)

    print(f"\n--- Processing for {instance_name} finished. Outputs in {INSTANCE_OUTPUT_DIR} ---")

if __name__ == "__main__":
    main() 