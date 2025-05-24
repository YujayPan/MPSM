# --------------------------------------------------------------------------
# 0. Imports
# --------------------------------------------------------------------------
import os
import time
import argparse
import logging
import csv
import torch
import numpy as np
import concurrent.futures # For parallel execution
from functools import partial
import math # Ensure math is imported
import importlib # Ensure importlib is imported
import gc # For memory release
from tqdm import tqdm # Import tqdm

# Project-specific imports
from utils import (load_dataset, get_env, VRP_DATA_FORMAT, DATASET_PATHS,
                   AverageMeter, TimeEstimator, seed_everything)
# from models import MOEModel, MOEModel_Light # Not directly used if using partition_instance
from partitioner_solver_utils import (
    load_moe_model, partition_instance, 
    create_subproblem_instance, pad_subproblem_batch, 
    prepare_batch_tensor_data, solve_vrp_batch, 
    DEFAULT_MODEL_PARAMS
)
# from ortools_solver import ortools_solve_vrp # OR-Tools not primary focus now

# --------------------------------------------------------------------------
# 1. Constants & Configuration
# --------------------------------------------------------------------------
PROBLEM_SIZES_TO_TEST = [50, 100, 200, 500, 1000, 2000, 5000]
DEFAULT_NUM_INSTANCES = 100 # Number of instances per size/type to test
DEFAULT_WORKERS = os.cpu_count() # Default workers for parallel execution
# List of all possible methods that could be tested
METHODS_TO_TEST = ['DirectSolver', 'DirectORTools', 'PartitionSolverParallel', 'PartitionORToolsParallel']

# Candidate adaptive merge target sizes for specific problem scales
CANDIDATE_ADAPTIVE_TARGETS = {
    200: [20, 25, 40, 50],
    500: [20, 25, 50, 100, 125],
    1000: [50, 100, 125, 200, 250],
    2000: [50, 80, 100, 125, 200, 250, 400, 500],
    5000: [50, 100, 125, 200, 250, 500, 1000, 1250],
}

# --------------------------------------------------------------------------
# 2. Argument Parser
# --------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="VRP Solver Comparison Framework")
    # --- Test Configuration ---
    parser.add_argument('--problems', nargs='+', default=["CVRP"], help="List of VRP problem types to test.") # Default to CVRP for quicker testing
    parser.add_argument('--sizes', nargs='+', type=int, default=[50, 100], help="List of problem sizes (N) to test.") # Default smaller sizes
    parser.add_argument('--num_instances', type=int, default=50, help="Number of instances to test per problem size/type.") # Default fewer instances
    parser.add_argument('--seed', type=int, default=2024, help="Random seed.")

    # --- Target Method for Optimization ---
    parser.add_argument('--target_method', type=str, nargs='+', default=['PartitionSolverParallel'], 
                        choices=METHODS_TO_TEST, 
                        help="Method(s) to run and potentially optimize (e.g., find best adaptive target). Choose one or more.")
    parser.add_argument('--test_batch_size', type=int, default=32, 
                        help="Batch size for processing instances during testing (currently only used by PartitionSolverParallel).")

    # --- Model Paths (Optional) ---
    parser.add_argument('--solver_checkpoint', type=str, default=None, # Not required
                        help="Path to the pre-trained Solver model checkpoint, OR 'ortools' to only use OR-Tools as the solver.")
    parser.add_argument('--partitioner_checkpoint', type=str, default=None, # Not required
                        help="Path to the pre-trained Partitioner model checkpoint. If None, partitioned methods are skipped.")

    # --- Solver Model Parameters (Optional, only needed if solver_checkpoint is a path) ---
    parser.add_argument('--solver_model_type', type=str, default="MOE",
                        help="Type of solver model arch used in checkpoint (e.g., MOE).")
    parser.add_argument('--solver_num_experts', type=int, default=4,
                        help="Number of experts for solver model (if MOE).")
    parser.add_argument('--solver_embedding_dim', type=int, default=128, help="(Optional) Embedding dimension for Solver model.")
    parser.add_argument('--solver_ff_hidden_dim', type=int, default=512, help="(Optional) Feed-forward hidden dim for Solver model.")
    parser.add_argument('--solver_encoder_layer_num', type=int, default=6, help="(Optional) Number of encoder layers for Solver model.")
    # Add other necessary solver model params as optional args...
    # parser.add_argument('--solver_embedding_dim', type=int, default=128)
    # ...

    # --- Partitioner Model Parameters (Optional, only needed if partitioner_checkpoint is provided) ---
    parser.add_argument('--partitioner_model_type', type=str, default="MOE",
                        help="Type of partitioner model arch used in checkpoint (e.g., MOE).")
    parser.add_argument('--partitioner_num_experts', type=int, default=4,
                        help="Number of experts for partitioner model (if MOE).")
    parser.add_argument('--partitioner_merge_num', type=int, default=-1, # Already existed, keep it
                        help="Merge number used by the partitioner. <=0 for adaptive merge.")
    parser.add_argument('--adaptive_merge_target_size', type=int, default=0, 
                        help="Target node count for adaptive merging by partitioner (if merge_num <=0). <=0 for dynamic selection.")
    # Add other necessary partitioner model params as optional args...
    parser.add_argument('--partitioner_embedding_dim', type=int, default=128, help="(Optional) Embedding dimension for Partitioner model.")
    parser.add_argument('--partitioner_ff_hidden_dim', type=int, default=512, help="(Optional) Feed-forward hidden dim for Partitioner model.")
    parser.add_argument('--partitioner_encoder_layer_num', type=int, default=6, help="(Optional) Number of encoder layers for Partitioner model.")
    # Add more partitioner params if needed...

    # --- Method Specific Parameters ---
    parser.add_argument('--ortools_timelimit', type=int, default=30, # Reduced default
                        help="Time limit in seconds for each OR-Tools solve call (direct or subproblem).")
    parser.add_argument('--solver_aug_factor', type=int, default=1,
                        help="Augmentation factor for Solver model inference.")
    # parser.add_argument('--solver_batch_size', type=int, default=128, help="Batch size for solver inference (if applicable).") # Less relevant for single instance tests

    # --- Execution Control ---
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help="Max number of worker processes for parallel subproblem solving.")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use if CUDA is enabled.")

    # --- Output ---
    parser.add_argument('--output_csv', type=str, default='comparison_results.csv', help="Path to save the CSV results.")
    parser.add_argument('--log_file', type=str, default='comparison_test.log', help="Path to save the log file.")
    parser.add_argument('--verbose_log', action='store_true', help="Print detailed logs (INFO level) to console if set.")

    args = parser.parse_args()
    return args

# --------------------------------------------------------------------------
# 3. Logging Setup
# --------------------------------------------------------------------------
def setup_logging(log_file, verbose=False):
    """ Configures logging to file (INFO) and console (WARNING or INFO). """
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    root_logger = logging.getLogger() 
    root_logger.setLevel(logging.INFO) # Set root logger level

    # Clear existing handlers (important if run multiple times in notebooks etc.)
    root_logger.handlers.clear()

    # File Handler (always INFO)
    try:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger at {log_file}: {e}")

    # Console Handler (level depends on verbosity)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    if verbose:
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.WARNING)
    root_logger.addHandler(console_handler)

# --------------------------------------------------------------------------
# 4. Helper Function for Batched PartitionSolverParallel
# --------------------------------------------------------------------------
def run_batched_partition_solve(
    batch_instance_tuples,
    problem_type,
    partitioner_model,
    partitioner_params,
    current_adaptive_target,
    solver_model,
    solver_env_class,
    device,
    args
):
    batch_size = len(batch_instance_tuples)
    if batch_size == 0:
        return []

    # --- Timing ---
    batch_partition_time_start = time.time()

    # --- Hybrid Partitioning: Per-instance call to partition_instance ---
    per_instance_raw_sequences_and_subs = [] # Stores (raw_seq_list, subproblem_tuples_for_instance)
    
    for inst_tuple in batch_instance_tuples:
        _sub_tuples_inst, _raw_seq_inst = partition_instance(
            original_instance_tuple=inst_tuple,
            problem_type=problem_type,
            partitioner_checkpoint_path=None, # Model is pre-loaded
            merge_num=args.partitioner_merge_num, 
            device=device,
            partitioner_model_params=partitioner_params, 
            partitioner_model=partitioner_model,
            target_node_count_for_merge=current_adaptive_target
        )
        per_instance_raw_sequences_and_subs.append(
            (_raw_seq_inst if _raw_seq_inst else [], _sub_tuples_inst if _sub_tuples_inst else [])
        )
    
    batch_partition_time = time.time() - batch_partition_time_start
    
    # --- Subproblem Collection ---
    batch_solving_time_start = time.time()
    all_subproblems_global = []
    instance_subproblem_metadata = [] # Stores (num_subproblems_for_this_orig_inst, start_idx_in_global)

    for i in range(batch_size):
        _raw_seq, sub_tuples_for_instance = per_instance_raw_sequences_and_subs[i]
        
        num_subs_this_instance = len(sub_tuples_for_instance)
        start_idx_global = len(all_subproblems_global)
        instance_subproblem_metadata.append((num_subs_this_instance, start_idx_global))
        
        if num_subs_this_instance > 0:
            all_subproblems_global.extend(sub_tuples_for_instance)
        # If num_subs_this_instance is 0, this instance effectively failed partitioning for solving.

    # --- Batched Solving of ALL collected subproblems ---
    flat_solver_results = None
    if all_subproblems_global:
        try:
            padded_global_subproblems, target_pad_size_global = pad_subproblem_batch(all_subproblems_global, problem_type)
            if not padded_global_subproblems:
                 raise ValueError("Failed to pad global subproblem batch for solving")
            tensor_data_global = prepare_batch_tensor_data(padded_global_subproblems, problem_type, device)
            if not tensor_data_global:
                raise ValueError("Failed to prepare tensor data for global subproblem batch for solving")
            
            solver_model.eval() # Ensure solver is in eval
            flat_solver_results = solve_vrp_batch(
                solver_model=solver_model,
                solver_env_class=solver_env_class,
                original_instance_tuples=all_subproblems_global, 
                padded_batch_data=tensor_data_global,
                padded_problem_size=target_pad_size_global,
                problem_type=problem_type, # Subproblems inherit type
                device=device,
                aug_factor=args.solver_aug_factor
            )
            if not flat_solver_results or len(flat_solver_results) != len(all_subproblems_global):
                logging.error(f"Global subproblem solver results mismatch. Expected {len(all_subproblems_global)}, got {len(flat_solver_results) if flat_solver_results else 0}.")
                flat_solver_results = None 
        except Exception as e:
            logging.error(f"Error solving global subproblem batch: {e}", exc_info=True)
            flat_solver_results = None
    
    batch_solving_time = time.time() - batch_solving_time_start

    # --- Aggregate results per original instance ---
    final_results_for_batch = []
    for i in range(batch_size):
        num_subproblems_this_orig_inst, start_idx_global_this_orig_inst = instance_subproblem_metadata[i]
        score_for_orig_instance = float('inf')

        if num_subproblems_this_orig_inst == 0: # Partitioning failed to produce solvable subproblems
            score_for_orig_instance = float('inf')
        elif flat_solver_results is None: # Global solving step failed
            score_for_orig_instance = float('inf')
        else:
            current_orig_inst_total_cost = 0
            # Extract the results for subproblems belonging to this original instance
            sub_results_for_this_orig_inst = flat_solver_results[start_idx_global_this_orig_inst : start_idx_global_this_orig_inst + num_subproblems_this_orig_inst]
            
            if len(sub_results_for_this_orig_inst) != num_subproblems_this_orig_inst:
                logging.error(f"Logic error: Sub-result slice for instance {i} incorrect.")
                current_orig_inst_total_cost = float('inf')
            else:
                for cost, _ in sub_results_for_this_orig_inst:
                    if cost == float('inf') or cost is None or math.isnan(cost):
                        current_orig_inst_total_cost = float('inf')
                        break 
                    current_orig_inst_total_cost += cost
            score_for_orig_instance = current_orig_inst_total_cost
        
        # Apportion times (simple average for now)
        # Note: batch_partition_time is total for batch, batch_solving_time is total for solving all subproblems
        # This timing might not be perfectly per-instance if partitioning times vary wildly.
        avg_inst_partition_time = batch_partition_time / batch_size if batch_size > 0 else 0
        # Solving time is trickier to apportion directly if subproblem counts differ vastly.
        # For now, use an average of the global solving time. More accurate would be complex.
        avg_inst_solving_time = batch_solving_time / batch_size if batch_size > 0 else 0


        final_results_for_batch.append({
            'score': score_for_orig_instance,
            'total_time_seconds': avg_inst_partition_time + avg_inst_solving_time,
            'partition_time_seconds': avg_inst_partition_time,
            'solving_time_seconds': avg_inst_solving_time,
            'num_subproblems': num_subproblems_this_orig_inst
        })
        
    return final_results_for_batch

# --------------------------------------------------------------------------
# 5. Main Execution Logic
# --------------------------------------------------------------------------
def main():
    args = parse_arguments()
    setup_logging(args.log_file, args.verbose_log)
    seed_everything(args.seed)

    # --- Unconditional Console Start Message ---
    run_id = f"TestRun_{time.strftime('%Y%m%d_%H%M%S')}" # Simple run id
    print(f"\n>>> Starting Test Run: {run_id} <<<" ) 
    print(f"    See detailed logs in: {args.log_file}")
    print(f"    Results will be saved to: {args.output_csv}")
    if not args.verbose_log: print(f"    Use --verbose_log for detailed console output.")
    print("---")
    # --- End Start Message ---

    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f'cuda:{args.gpu_id}')
        logging.info(f"Using GPU: {args.gpu_id}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
        args.gpu_id = None # Ensure gpu_id is None if using CPU

    # --- Determine Methods to Run ---
    requested_methods = args.target_method # This is now a list
    methods_to_run = []
    
    # Check resource availability
    use_solver_model = args.solver_checkpoint is not None and args.solver_checkpoint.lower() != 'ortools'
    use_partitioner = args.partitioner_checkpoint is not None
    ortools_available = False
    try:
        import ortools_solver # Attempt import
        ortools_solve_vrp = getattr(ortools_solver, 'ortools_solve_vrp')
        ortools_available = True
    except (ImportError, AttributeError): pass
    
    # Filter requested methods based on availability
    if 'DirectSolver' in requested_methods:
        if use_solver_model:
            methods_to_run.append('DirectSolver')
        else: logging.warning("Skipping DirectSolver (requested, but no solver checkpoint provided).")
            
    if 'DirectORTools' in requested_methods:
        if ortools_available:
            methods_to_run.append('DirectORTools')
        else: logging.warning("Skipping DirectORTools (requested, but OR-Tools not available).")
            
    if 'PartitionSolverParallel' in requested_methods:
        if use_partitioner and use_solver_model:
            methods_to_run.append('PartitionSolverParallel')
        else: logging.warning("Skipping PartitionSolverParallel (requested, but partitioner or solver checkpoint missing).")
            
    if 'PartitionORToolsParallel' in requested_methods:
        if use_partitioner and ortools_available:
            methods_to_run.append('PartitionORToolsParallel')
        else: logging.warning("Skipping PartitionORToolsParallel (requested, but partitioner or OR-Tools missing).")
    
    if not methods_to_run:
        logging.error("No requested methods can be run based on provided arguments and available modules. Exiting.")
        return

    logging.info(f"Methods available and requested to run: {methods_to_run}")

    # --- Load Models --- 
    logging.info("Loading models...")
    solver_model = None
    solver_params = None
    partitioner_model = None
    partitioner_params = None
    CurrentEnvClass = None # Define here to ensure scope

    try:
        # --- Load Solver Model ---
        solver_params = DEFAULT_MODEL_PARAMS.copy()
        solver_params['model_type'] = args.solver_model_type
        solver_params['num_experts'] = args.solver_num_experts
        solver_params['problem'] = args.problems[0] if args.problems else "CVRP"
        solver_params['device'] = device
        # Override defaults with specific args if provided
        if args.solver_embedding_dim is not None: solver_params['embedding_dim'] = args.solver_embedding_dim
        if args.solver_ff_hidden_dim is not None: solver_params['ff_hidden_dim'] = args.solver_ff_hidden_dim
        if args.solver_encoder_layer_num is not None: solver_params['encoder_layer_num'] = args.solver_encoder_layer_num
        # Add overrides for other args as needed
        
        solver_model = load_moe_model(args.solver_checkpoint, device, model_type=args.solver_model_type, model_params=solver_params)
        if not solver_model: raise ValueError(f"Failed to load Solver model from {args.solver_checkpoint}")
        solver_model.eval()
        logging.info(f"Solver Checkpoint loaded: {args.solver_checkpoint}")

        # --- Load Partitioner Model ---
        partitioner_params = DEFAULT_MODEL_PARAMS.copy()
        partitioner_params['model_type'] = args.partitioner_model_type
        partitioner_params['num_experts'] = args.partitioner_num_experts
        partitioner_params['problem'] = args.problems[0] if args.problems else "CVRP" # Use first problem type
        partitioner_params['device'] = device
        # Override defaults with specific args if provided
        if args.partitioner_embedding_dim is not None: partitioner_params['embedding_dim'] = args.partitioner_embedding_dim
        if args.partitioner_ff_hidden_dim is not None: partitioner_params['ff_hidden_dim'] = args.partitioner_ff_hidden_dim
        if args.partitioner_encoder_layer_num is not None: partitioner_params['encoder_layer_num'] = args.partitioner_encoder_layer_num
        # Add overrides for other args as needed

        partitioner_model = load_moe_model(args.partitioner_checkpoint, device, model_type=args.partitioner_model_type, model_params=partitioner_params)
        if not partitioner_model: raise ValueError(f"Failed to load Partitioner model from {args.partitioner_checkpoint}")
        partitioner_model.eval()
        logging.info(f"Partitioner Checkpoint loaded: {args.partitioner_checkpoint}")

    except Exception as e:
        logging.error(f"Failed during model/parameter preparation: {e}", exc_info=True)
        return

    # --- Prepare CSV Output ---
    csv_headers = ['problem_type', 'problem_size', 'instance_index', 'method',
                   'score', 'total_time_seconds',
                   'partition_time_seconds', 'solving_time_seconds', 'num_subproblems',
                   'current_adaptive_target']
    # Removed all_results list as we write directly

    logging.info("Starting comparison test for PartitionSolverParallel...")
    global_start_time = time.time()

    # --- Test Loop --- 
    csvfile = None
    try:
        csvfile = open(args.output_csv, 'w', newline='')
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()

        for problem_type in args.problems:
            logging.info(f"--- Testing Problem Type: {problem_type} ---")
            # Get Env Class once per problem type
            try:
                EnvClassList = get_env(problem_type)
                if not EnvClassList: raise ValueError(f"No EnvClass found for {problem_type}")
                CurrentEnvClass = EnvClassList[0]
                logging.info(f"Using Env Class: {CurrentEnvClass.__name__}")
            except Exception as e:
                logging.error(f"Failed to get Env class for {problem_type}: {e}. Skipping.")
                continue # Skip to next problem type

            for size in args.sizes:
                logging.info(f"--- Testing Size: {size} ---")
                # Load dataset
                full_dataset = None
                try:
                    if problem_type not in DATASET_PATHS or size not in DATASET_PATHS[problem_type]:
                        logging.warning(f"Dataset path definition not found for {problem_type} size {size}. Skipping size.")
                        continue
                    dataset_info = DATASET_PATHS[problem_type][size]
                    dataset_path = dataset_info['data']
                    if not os.path.exists(dataset_path):
                        script_dir = os.path.dirname(__file__) if "__file__" in locals() else "."
                        relative_path = os.path.join(script_dir, dataset_path)
                        if os.path.exists(relative_path): dataset_path = relative_path
                        else: raise FileNotFoundError(f"Dataset file not found at {dataset_info['data']} or {relative_path}")
                    full_dataset = load_dataset(dataset_path)
                    num_available_instances = len(full_dataset)
                    instances_to_run = min(args.num_instances, num_available_instances)
                    if instances_to_run < args.num_instances: logging.warning(f"Requested {args.num_instances} instances, but only {num_available_instances} available.")
                    if instances_to_run <= 0: logging.warning(f"No instances to run. Skipping."); continue
                    logging.info(f"Testing {instances_to_run} instances from {dataset_path}")
                except Exception as e:
                    logging.error(f"Failed to load dataset for {problem_type} N={size}: {e}. Skipping size.")
                    continue
                
                # Determine candidate target sizes for this problem size
                adaptive_targets_for_this_size = CANDIDATE_ADAPTIVE_TARGETS.get(size, [args.adaptive_merge_target_size])
                # Force adaptive if merge_num is set accordingly, ignore candidates otherwise
                is_adaptive_run = args.partitioner_merge_num <= 0 
                if not is_adaptive_run:
                    adaptive_targets_for_this_size = [args.adaptive_merge_target_size] # Use CLI value once
                    logging.info(f"-- Using Fixed Merge Num: {args.partitioner_merge_num} --")
                
                # Loop through candidates (or single default if fixed merge)
                for current_target_candidate in adaptive_targets_for_this_size:
                    if is_adaptive_run:
                        logging.info(f"-- Testing Adaptive Merge Target: {current_target_candidate} --")
                    
                    # --- Determine Effective Batch Size for this Size --- 
                    base_batch_size = args.test_batch_size
                    cap_batch_size = base_batch_size # Default
                    
                    # Apply caps similar to train_partitioner generalization validation
                    if size >= 5000:
                        cap_batch_size = 4
                    elif size >= 2000:
                        cap_batch_size = 8 # Example cap for N=2000
                    elif size >= 1000:
                        cap_batch_size = 16 # Example cap for N=1000
                    # Add more elif conditions here for other sizes if needed
                    
                    effective_test_batch_size = min(base_batch_size, cap_batch_size)
                    # Ensure batch size is at least 1
                    effective_test_batch_size = max(1, effective_test_batch_size)
                    
                    if effective_test_batch_size < base_batch_size:
                        logging.info(f"Testing N={size}: Dynamically capping test batch size from {base_batch_size} to {effective_test_batch_size}.")
                    
                    # --- End Determine Effective Batch Size ---

                    # Process instances in batches using the effective size
                    num_batches = math.ceil(instances_to_run / effective_test_batch_size)
                    logging.info(f"Processing {instances_to_run} instances in {num_batches} batches (effective_size={effective_test_batch_size}).")

                    # --- Wrap batch loop with tqdm for console progress ---
                    pbar_batches = tqdm(range(num_batches), desc=f"N={size} Target={current_target_candidate}", unit="batch", leave=True)
                    for batch_idx in pbar_batches:
                    # --- End tqdm wrap ---
                    
                        batch_start_idx = batch_idx * effective_test_batch_size
                        batch_end_idx = min(batch_start_idx + effective_test_batch_size, instances_to_run)
                        current_batch_indices = list(range(batch_start_idx, batch_end_idx))
                        current_batch_instance_tuples = [full_dataset[i] for i in current_batch_indices]
                        current_actual_batch_size = len(current_batch_instance_tuples)

                        if current_actual_batch_size == 0:
                            continue
                        
                        logging.info(f"Processing batch {batch_idx+1}/{num_batches} (Size: {current_actual_batch_size}, Indices: {current_batch_indices[0]}-{current_batch_indices[-1]})...")
                        
                        # Call the batched processing function
                        batch_results = run_batched_partition_solve(
                            batch_instance_tuples=current_batch_instance_tuples,
                            problem_type=problem_type,
                            partitioner_model=partitioner_model,
                            partitioner_params=partitioner_params,
                            current_adaptive_target=current_target_candidate,
                            solver_model=solver_model,
                            solver_env_class=CurrentEnvClass,
                            device=device,
                            args=args
                        )
                        
                        # Write results for each instance in the batch
                        for k, result_dict in enumerate(batch_results):
                            instance_index_in_dataset = current_batch_indices[k]
                            writer.writerow({
                                'problem_type': problem_type,
                                'problem_size': size,
                                'instance_index': instance_index_in_dataset,
                                'method': 'PartitionSolverParallel', # Only method running
                                'score': f"{result_dict['score']:.4f}" if result_dict['score'] != float('inf') else 'inf',
                                'total_time_seconds': f"{result_dict['total_time_seconds']:.4f}",
                                'partition_time_seconds': f"{result_dict['partition_time_seconds']:.4f}",
                                'solving_time_seconds': f"{result_dict['solving_time_seconds']:.4f}",
                                'num_subproblems': result_dict['num_subproblems'],
                                'current_adaptive_target': current_target_candidate if is_adaptive_run else 'N/A'
                            })
                        
                        csvfile.flush() # Flush after each batch
                        # Optional: Add memory release here if needed between batches
                        if use_cuda: torch.cuda.empty_cache(); gc.collect()
                        
                        # Update tqdm description (optional)
                        pbar_batches.set_postfix_str(f"Last Batch: {current_actual_batch_size} instances", refresh=True)

                logging.info(f"--- Finished Size: {size} ---")
            logging.info(f"--- Finished Problem Type: {problem_type} ---")

    except IOError as e:
        logging.error(f"Error opening or writing CSV file {args.output_csv}: {e}")
    except Exception as e:
         logging.error(f"An unexpected error occurred during testing: {e}", exc_info=True)
    finally:
         if csvfile:
             csvfile.close()
             logging.info(f"Results file closed: {args.output_csv}")

    global_end_time = time.time()
    logging.info(f"Comparison test finished. Total time: {global_end_time - global_start_time:.2f} seconds.")
    
    # --- Unconditional Console End Message ---
    print("---")
    print(f">>> Finished Test Run: {run_id} <<<" )
    print(f"    Results saved to: {args.output_csv}")
    print(f"    Detailed logs in: {args.log_file}")
    # --- End Message ---

# --------------------------------------------------------------------------
# 6. Script Execution Guard
# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
