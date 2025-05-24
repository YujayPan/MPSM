# --------------------------------------------------------------------------
# Script: generate_partitions.py
# Description: Generates partitions from raw VRP instances and saves them.
# --------------------------------------------------------------------------

# 0. Imports
# --------------------------------------------------------------------------
import os
import time
import argparse
import logging
import pickle
import json
import torch
import numpy as np
import math # Ensure math is imported
import gc # For memory release
from tqdm import tqdm # Import tqdm
import importlib # Ensure importlib is imported

# Project-specific imports
from utils import (load_dataset, get_env, VRP_DATA_FORMAT, DATASET_PATHS,
                   AverageMeter, TimeEstimator, seed_everything)
from partitioner_solver_utils import (
    load_moe_model, partition_instance, 
    merge_subproblems_by_centroid_fixed_size, 
    DEFAULT_MODEL_PARAMS
)

# --------------------------------------------------------------------------
# 1. Constants & Configuration
# --------------------------------------------------------------------------
PROBLEM_SIZES_TO_TEST = [50, 100, 200, 500, 1000, 2000, 5000]
SUPPORTED_PROBLEM_TYPES = ["CVRP", "OVRP", "VRPB", "VRPL", "VRPTW"] # Matching test_comparison.py
DEFAULT_NUM_INSTANCES = 100
DEFAULT_OUTPUT_DIR = 'partition_results'
DEFAULT_MERGE_CONFIGS = ['raw_subroutes', 'm1', 'm3', 'adaptive'] # Default to only saving initial split by zeros

# --------------------------------------------------------------------------
# 2. Argument Parser
# --------------------------------------------------------------------------
def parse_arguments_partitioner():
    parser = argparse.ArgumentParser(description="VRP Instance Partitioner Framework")
    # --- Dataset & Instance Selection ---
    parser.add_argument('--problems', nargs='+', default=SUPPORTED_PROBLEM_TYPES, 
                        help=f"List of VRP problem types to partition. Default: {SUPPORTED_PROBLEM_TYPES}")
    parser.add_argument('--sizes', nargs='+', type=int, default=PROBLEM_SIZES_TO_TEST, 
                        help=f"List of problem sizes (N) to partition. Default: {PROBLEM_SIZES_TO_TEST}")
    parser.add_argument('--num_instances', type=int, default=DEFAULT_NUM_INSTANCES, 
                        help=f"Number of instances to partition per problem size/type. Default: {DEFAULT_NUM_INSTANCES}")
    parser.add_argument('--seed', type=int, default=2024, help="Random seed.")

    # --- Partitioner Model (Fine-tuned Solver) ---
    parser.add_argument('--partitioner_checkpoint', type=str, required=True,
                        help="Path to the pre-trained Partitioner model checkpoint (fine-tuned Solver).")
    # Parameters for loading the partitioner model (copied from test_comparison.py)
    parser.add_argument('--partitioner_model_type', type=str, default="MOE", help="Partitioner model architecture type.")
    parser.add_argument('--partitioner_num_experts', type=int, default=4, help="Number of experts for Partitioner model (if MOE).")
    parser.add_argument('--partitioner_embedding_dim', type=int, default=128, help="Embedding dimension for Partitioner model.")
    parser.add_argument('--partitioner_ff_hidden_dim', type=int, default=512, help="Feed-forward hidden dim for Partitioner model.")
    parser.add_argument('--partitioner_encoder_layer_num', type=int, default=6, help="Number of encoder layers for Partitioner model.")

    # --- Output & Merging ---
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Root directory to save partition results. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument('--merge_configs', nargs='+', type=str, default=DEFAULT_MERGE_CONFIGS,
                        help="List of merge configurations to apply and save. "
                             "Examples: 'raw_subroutes' (initial split), 'm1', 'm3', 'adaptive_s50', 'adaptive_s100'. "
                             f"Default: {DEFAULT_MERGE_CONFIGS}")
    
    # --- Execution Control ---
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use if CUDA is enabled.")

    # --- Logging ---
    parser.add_argument('--log_file', type=str, default='generate_partitions.log', 
                        help="Name of the log file to be saved within the run-specific output directory.")
    parser.add_argument('--verbose_log', action='store_true', help="Print detailed logs (INFO level) to console if set.")

    args = parser.parse_args()
    return args

# --------------------------------------------------------------------------
# 3. Logging Setup
# --------------------------------------------------------------------------
def setup_logging(log_file_path, verbose=False):
    """ Configures logging to file (INFO) and console (WARNING or INFO). """
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    root_logger = logging.getLogger() 
    root_logger.setLevel(logging.INFO)

    root_logger.handlers.clear()

    try:
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger at {log_file_path}: {e}")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    if verbose:
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.WARNING)
    root_logger.addHandler(console_handler)

# --------------------------------------------------------------------------
# 4. Main Partitioning Logic
# --------------------------------------------------------------------------
def main_partitioner():
    args = parse_arguments_partitioner()

    # --- Setup Output Paths (Run-specific) --- 
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_id = f"PartitionGen_{run_timestamp}"
    # Base output for this run will be args.output_dir / run_id
    # Specific instance data will be further nested.
    # Log file will be at the root of this run's output directory.
    run_output_dir = os.path.join(args.output_dir, run_id)
    try:
        os.makedirs(run_output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating base run output directory {run_output_dir}: {e}. Using current directory as fallback for log.")
        run_output_dir = "." 
        
    full_log_path = os.path.join(run_output_dir, args.log_file)
    
    setup_logging(full_log_path, args.verbose_log)
    seed_everything(args.seed)

    # --- Unconditional Console Start Message ---
    print(f"\n>>> Starting Partition Generation Run: {run_id} <<<")
    print(f"    Output Root: {args.output_dir}")
    print(f"    Run Output Dir: {run_output_dir}") # For instance-specific data
    print(f"    Logs: {full_log_path}")
    if not args.verbose_log: print(f"    Use --verbose_log for detailed console output.")
    print("---")

    logging.info(f"Partition Generation Run ID: {run_id}")
    logging.info(f"Script arguments: {args}")

    # --- Setup Device ---
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu_id}' if use_cuda else 'cpu')
    logging.info(f"Using device: {device}")

    # --- Load Partitioner Model (once) ---
    partitioner_model = None
    partitioner_params = None
    logging.info("Loading Partitioner model (fine-tuned Solver)...")
    try:
        partitioner_params = DEFAULT_MODEL_PARAMS.copy() # Start with defaults
        partitioner_params['model_type'] = args.partitioner_model_type
        partitioner_params['num_experts'] = args.partitioner_num_experts
        partitioner_params['device'] = device # Ensure device is in params for model
        if args.partitioner_embedding_dim is not None: partitioner_params['embedding_dim'] = args.partitioner_embedding_dim
        if args.partitioner_ff_hidden_dim is not None: partitioner_params['ff_hidden_dim'] = args.partitioner_ff_hidden_dim
        if args.partitioner_encoder_layer_num is not None: partitioner_params['encoder_layer_num'] = args.partitioner_encoder_layer_num
        
        # The 'problem' param for load_moe_model usually comes from the checkpoint's saved args.
        # If not found, it might use a default from DEFAULT_MODEL_PARAMS.
        # For partitioning, the actual problem type of the instance being partitioned is more relevant
        # for the environment, not strictly for model loading if architecture is fixed.
        
        partitioner_model = load_moe_model(
            args.partitioner_checkpoint, 
            device, 
            model_type=args.partitioner_model_type, 
            model_params=partitioner_params
        )
        if not partitioner_model: 
            raise ValueError("Partitioner model loading returned None.")
        partitioner_model.eval()
        logging.info(f"Partitioner model loaded successfully from {args.partitioner_checkpoint}")
    except Exception as e:
        logging.error(f"Failed to load Partitioner model: {e}", exc_info=True)
        print(f"Error: Critical failure loading partitioner model. Exiting.")
        return

    # --- Main Loop for Partition Generation ---
    logging.info("Starting partition generation process...")
    
    # Helper function to parse merge configuration strings
    def parse_merge_config_string(config_name: str):
        if config_name == 'raw_subroutes': # Should be handled separately
            return None, None 
        elif config_name == 'm1':
            return 1, 0
        elif config_name == 'm3':
            return 3, 0
        elif config_name == 'adaptive':
            return -1, 0 # Dynamic target selection based on problem size
        elif config_name.startswith('adaptive_s'):
            try:
                target_size = int(config_name.split('adaptive_s')[-1])
                return -1, target_size
            except ValueError:
                logging.warning(f"Could not parse adaptive target size from '{config_name}'. Using dynamic selection.")
                return -1, 0
        else:
            logging.warning(f"Unknown merge configuration name: '{config_name}'. Skipping.")
            return None, None

    # Helper function to generate raw sequence and initial subproblem node lists
    # This function encapsulates the model rollout and initial splitting logic
    # Adapted from parts of partitioner_solver_utils.partition_instance
    def generate_sequence_and_initial_routes(
        original_instance_tuple, 
        problem_type, 
        loaded_partitioner_model, # Expects pre-loaded model
        device_obj, # Expects torch.device object
        max_seq_len_factor=2
    ):
        from partitioner_solver_utils import _split_sequence_by_zeros # Local import for clarity
        from utils import get_env # Local import for clarity
        from partitioner_solver_utils import pad_subproblem_batch, prepare_batch_tensor_data # Local imports

        raw_sequence = None
        initial_subproblem_node_lists = []

        try:
            # Determine Instance Size
            node_xy_index = VRP_DATA_FORMAT[problem_type].index('node_xy')
            num_customer_nodes = len(original_instance_tuple[node_xy_index])
            if num_customer_nodes <= 0:
                logging.error("Instance has no customer nodes.")
                return None, []
            
            max_seq_len = max_seq_len_factor * (num_customer_nodes + 1)

            # Setup Environment for Rollout
            EnvClassList = get_env(problem_type)
            if not EnvClassList:
                logging.error(f"Could not get env class for {problem_type}")
                return None, []
            PartitionEnvClass = EnvClassList[0]
            
            # For partitioning, pomo_size is 1, problem_size is num_customer_nodes
            env_params = {"problem_size": num_customer_nodes, "pomo_size": 1, "device": device_obj}
            partition_env = PartitionEnvClass(**env_params)
            
            # Pad instance to its own size (no actual padding, just formatting for env)
            # The pad_subproblem_batch expects a list of tuples
            padded_batch_tuples, target_pad_size = pad_subproblem_batch(
                [original_instance_tuple], problem_type, num_customer_nodes
            )
            if not padded_batch_tuples or target_pad_size != num_customer_nodes:
                logging.error(f"Padding/Target size mismatch. Expected {num_customer_nodes}, got {target_pad_size}")
                return None, []
            
            instance_tensor_data = prepare_batch_tensor_data(padded_batch_tuples, problem_type, device_obj)
            if not instance_tensor_data:
                logging.error("Failed to prepare instance tensor data for partitioning.")
                return None, []
            
            partition_env.load_problems(batch_size=1, problems=instance_tensor_data, aug_factor=1)

            # Generate Partition Sequence via Rollout
            loaded_partitioner_model.eval()
            loaded_partitioner_model.set_eval_type('argmax') # Deterministic for partitioning

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
                    logging.warning(f"Sequence generation reached max length ({max_seq_len}) for instance.")
                
                if hasattr(partition_env, 'selected_node_list') and partition_env.selected_node_list is not None:
                    if partition_env.selected_node_list.numel() > 0:
                        raw_sequence = partition_env.selected_node_list.view(-1).cpu().tolist()
                    else:
                        logging.warning("partition_env.selected_node_list is empty.")
                else:
                    logging.warning("partition_env.selected_node_list not found.")

            if raw_sequence is None:
                logging.error("Failed to generate raw sequence from environment rollout.")
                return None, []
            
            initial_subproblem_node_lists = _split_sequence_by_zeros(raw_sequence)

        except Exception as e:
            logging.error(f"Error in generate_sequence_and_initial_routes: {e}", exc_info=True)
            return None, []
            
        return raw_sequence, initial_subproblem_node_lists

    total_instances_processed = 0
    global_start_time = time.time()

    # --- Problem Type Loop ---
    for problem_type in args.problems:
        logging.info(f"--- Processing Problem Type: {problem_type} ---")
        if problem_type not in DATASET_PATHS:
            logging.warning(f"Dataset path definition not found for problem type {problem_type}. Skipping.")
            continue

        # --- Problem Size Loop ---
        for size in args.sizes:
            logging.info(f"--- Processing Size: {size} for {problem_type} ---")
            if size not in DATASET_PATHS[problem_type]:
                logging.warning(f"Dataset path definition not found for {problem_type} size {size}. Skipping size.")
                continue

            dataset_info = DATASET_PATHS[problem_type][size]
            dataset_path = dataset_info['data']
            
            if not os.path.exists(dataset_path):
                script_dir = os.path.dirname(__file__) if "__file__" in locals() else "."
                relative_path_check = os.path.join(script_dir, dataset_path)
                if os.path.exists(relative_path_check):
                    dataset_path = relative_path_check
                else:
                    logging.warning(f"Dataset file not found at '{dataset_path}' or '{relative_path_check}'. Skipping size {size} for {problem_type}.")
                    continue
            
            try:
                full_dataset_for_size = load_dataset(dataset_path)
                num_available_instances = len(full_dataset_for_size)
                instances_to_run_count = min(args.num_instances, num_available_instances)

                if instances_to_run_count <= 0:
                    logging.info(f"No instances to process for {problem_type} N={size}. Skipping.")
                    continue
                logging.info(f"Processing {instances_to_run_count} instances for {problem_type} N={size} from {dataset_path}")
                
                instances_to_process_all = full_dataset_for_size[:instances_to_run_count]

            except FileNotFoundError:
                logging.error(f"Dataset file {dataset_path} not found. Skipping size {size} for {problem_type}.")
                continue
            except Exception as e:
                logging.error(f"Failed to load dataset for {problem_type} N={size} from {dataset_path}: {e}", exc_info=True)
                continue

            # --- Instance Loop (with tqdm) ---
            pbar_instances = tqdm(range(instances_to_run_count), desc=f"{problem_type}_N{size}", unit="instance", leave=True)
            for instance_idx_in_dataset in pbar_instances:
                original_instance_tuple = instances_to_process_all[instance_idx_in_dataset]
                pbar_instances.set_postfix_str(f"Instance {instance_idx_in_dataset+1}/{instances_to_run_count}")

                instance_output_dir = os.path.join(run_output_dir, problem_type, str(size), str(instance_idx_in_dataset))
                try:
                    os.makedirs(instance_output_dir, exist_ok=True)
                except OSError as e:
                    logging.error(f"Could not create output directory for instance: {instance_output_dir}. Error: {e}. Skipping instance.")
                    continue
                
                # --- Save instance_info.json ---
                instance_info = {
                    "original_dataset_path": dataset_path,
                    "original_instance_index": instance_idx_in_dataset,
                    "problem_type": problem_type,
                    "size": size,
                    "partitioner_checkpoint_used": args.partitioner_checkpoint,
                    "partition_run_id": run_id # Save the main run_id for traceability
                }
                try:
                    with open(os.path.join(instance_output_dir, "instance_info.json"), 'w') as f_info:
                        json.dump(instance_info, f_info, indent=4)
                except IOError as e:
                    logging.error(f"Failed to write instance_info.json for instance {instance_idx_in_dataset}: {e}. Skipping instance.")
                    continue

                # --- 1. Generate raw sequence and initial split ---
                partition_time_start = time.time()
                raw_sequence, initial_subproblem_node_lists = generate_sequence_and_initial_routes(
                    original_instance_tuple,
                    problem_type,
                    partitioner_model,
                    device
                )
                partition_time_seconds = time.time() - partition_time_start

                if raw_sequence is None or initial_subproblem_node_lists is None:
                    logging.error(f"Failed to generate raw sequence/initial split for {problem_type} N{size} instance {instance_idx_in_dataset}. Skipping.")
                    # Optionally save a failure log here
                    continue
                
                # --- Save partition_log.json (raw_sequence, partition_time_seconds) ---
                partition_log_data = {
                    "raw_sequence_length": len(raw_sequence), # Save length to avoid huge logs if needed
                    "raw_sequence_preview": raw_sequence[:50] + (['...'] if len(raw_sequence) > 50 else []), # Preview
                    # "raw_sequence": raw_sequence, # Potentially very large, consider if needed
                    "initial_num_subproblems": len(initial_subproblem_node_lists),
                    "partition_time_seconds": round(partition_time_seconds, 4)
                }
                try:
                    with open(os.path.join(instance_output_dir, "partition_log.json"), 'w') as f_plog:
                        json.dump(partition_log_data, f_plog, indent=4)
                    # Save full raw sequence separately if too large for JSON log
                    with open(os.path.join(instance_output_dir, "raw_sequence.pkl"), 'wb') as f_raw_seq_pkl:
                        pickle.dump(raw_sequence, f_raw_seq_pkl)

                except IOError as e:
                    logging.error(f"Failed to write partition_log.json or raw_sequence.pkl for instance {instance_idx_in_dataset}: {e}")
                    # Continue processing merge configs if possible, but log the error

                # --- Handle 'raw_subroutes' config ---
                if 'raw_subroutes' in args.merge_configs:
                    try:
                        with open(os.path.join(instance_output_dir, "raw_subroutes.pkl"), 'wb') as f_raw_pkl:
                            pickle.dump(initial_subproblem_node_lists, f_raw_pkl)
                        logging.debug(f"Saved initial_subproblem_node_lists for instance {instance_idx_in_dataset} ({len(initial_subproblem_node_lists)} lists).")
                    except IOError as e:
                        logging.error(f"Failed to write raw_subroutes.pkl for instance {instance_idx_in_dataset}: {e}")

                # --- Prepare data needed for merge_subproblems_by_centroid_fixed_size ---
                # This data comes from the original instance tuple
                try:
                    node_xy_idx = VRP_DATA_FORMAT[problem_type].index('node_xy')
                    depot_xy_idx = VRP_DATA_FORMAT[problem_type].index('depot_xy')
                    original_loc_np = np.array(original_instance_tuple[node_xy_idx])
                    original_depot_np = np.array(original_instance_tuple[depot_xy_idx]).flatten() # Ensure depot is 1D
                except (IndexError, KeyError) as e:
                    logging.error(f"Could not extract node_xy/depot_xy from original instance tuple for merging: {e}. Skipping merge operations for this instance.")
                    continue # Skip merge configs for this instance if essential data is missing

                # --- Process other merge configurations ---
                for merge_config_name in args.merge_configs:
                    if merge_config_name == 'raw_subroutes':
                        continue # Already handled

                    parsed_merge_num, parsed_adaptive_target = parse_merge_config_string(merge_config_name)
                    if parsed_merge_num is None: # Parsing failed or unknown config
                        continue

                    merge_time_start = time.time()
                    try:
                        merged_node_lists = merge_subproblems_by_centroid_fixed_size(
                            initial_subproblems=initial_subproblem_node_lists,
                            original_loc=original_loc_np,
                            original_depot=original_depot_np,
                            problem_size_for_dynamic_target=size,
                            merge_num=parsed_merge_num,
                            target_node_count=parsed_adaptive_target,
                            problem_type=problem_type.upper()
                        )
                    except Exception as merge_exec_err:
                        logging.error(f"Error during merge_subproblems_by_centroid_fixed_size for config '{merge_config_name}': {merge_exec_err}", exc_info=True)
                        merged_node_lists = [] # Ensure it's an empty list on error
                        
                    merge_time_seconds = time.time() - merge_time_start

                    # Save merged_node_lists to merged_nodes_{merge_config_name}.pkl
                    merged_output_pkl_path = os.path.join(instance_output_dir, f"merged_nodes_{merge_config_name}.pkl")
                    try:
                        with open(merged_output_pkl_path, 'wb') as f_merged_pkl:
                            pickle.dump(merged_node_lists, f_merged_pkl)
                        logging.debug(f"Saved {merge_config_name} ({len(merged_node_lists)} lists) to {merged_output_pkl_path}")
                    except IOError as e:
                        logging.error(f"Failed to write {merged_output_pkl_path}: {e}")
                    
                    # Save merged_log_{merge_config_name}.json
                    merge_log_data = {
                        "merge_config_name": merge_config_name,
                        "num_merged_subproblems": len(merged_node_lists),
                        "merge_time_seconds": round(merge_time_seconds, 4)
                    }
                    merged_log_json_path = os.path.join(instance_output_dir, f"merged_log_{merge_config_name}.json")
                    try:
                        with open(merged_log_json_path, 'w') as f_mlog:
                            json.dump(merge_log_data, f_mlog, indent=4)
                    except IOError as e:
                        logging.error(f"Failed to write {merged_log_json_path}: {e}")
                
                total_instances_processed +=1
                # --- Memory cleanup after each instance ---
                if use_cuda:
                    torch.cuda.empty_cache()
                gc.collect()
                logging.debug(f"Memory cleanup after instance {instance_idx_in_dataset} for {problem_type} N={size}")
            
            pbar_instances.close()
            logging.info(f"--- Finished processing Size: {size} for {problem_type} ---")
        logging.info(f"--- Finished processing Problem Type: {problem_type} ---")
    
    global_end_time = time.time()
    logging.info(f"Partition generation finished. Processed {total_instances_processed} total instances.")
    logging.info(f"Total time: {global_end_time - global_start_time:.2f} seconds.")
    
    logging.info(f"Partition generation run {run_id} finished.")
    print(f"--- Finished Partition Generation Run: {run_id} ---")
    print(f"    Results saved under: {run_output_dir}")

# --------------------------------------------------------------------------
# 5. Script Execution Guard
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Set multiprocessing start method if relevant (though this script might not use ProcessPoolExecutor directly for now)
    # import multiprocessing as mp
    # if mp.get_start_method(allow_none=True) != 'spawn':
    #     try:
    #         mp.set_start_method('spawn', force=True)
    #         print(f"INFO: Multiprocessing start method set to 'spawn'.")
    #     except RuntimeError as e:
    #         print(f"WARNING: Could not set multiprocessing start method to 'spawn': {e}.")
            
    main_partitioner() 