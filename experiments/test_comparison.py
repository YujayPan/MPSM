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
SUPPORTED_PROBLEM_TYPES = ["CVRP", "OVRP", "VRPB", "VRPL", "VRPTW"]

# Define the different methods for the ablation study
ALL_METHODS = [
    'DirectSolver', 
    'DirectORTools', 
    'PartitionSolver_m1', # Partition+Solver, merge_num=1
    'PartitionSolver_m3', # Partition+Solver, merge_num=3
    'PartitionSolver_adaptive', # Partition+Solver, merge_num=-1 (dynamic target)
    'PartitionORTools_m1', # Partition+OR-Tools, merge_num=1
    'PartitionORTools_m3', # Partition+OR-Tools, merge_num=3
    'PartitionORTools_adaptive', # Partition+OR-Tools, merge_num=-1 (dynamic target)
]

ALL_SOLVER_METHODS = ['DirectSolver', 'PartitionSolver_m1', 'PartitionSolver_m3', 'PartitionSolver_adaptive']
ALL_ORTOOLS_METHODS = ['DirectORTools', 'PartitionORTools_m1', 'PartitionORTools_m3', 'PartitionORTools_adaptive']
    
ALL_PARTITION_SOLVER_METHODS = ['PartitionSolver_m1', 'PartitionSolver_m3', 'PartitionSolver_adaptive']
ALL_PARTITION_ORTOOLS_METHODS = ['PartitionORTools_m1', 'PartitionORTools_m3', 'PartitionORTools_adaptive']
ALL_PARTITION_METHODS = ALL_PARTITION_SOLVER_METHODS + ALL_PARTITION_ORTOOLS_METHODS

ALL_M1_METHODS = ['PartitionSolver_m1', 'PartitionORTools_m1']
ALL_M3_METHODS = ['PartitionSolver_m3', 'PartitionORTools_m3']
ALL_ADAPTIVE_METHODS = ['PartitionSolver_adaptive', 'PartitionORTools_adaptive']

# Hardcoded OR-Tools timelimit map based on problem size
ORTOOLS_TIMELIMIT_MAP = {
    50: 20,
    100: 30,
    200: 60,
    500: 60,
    1000: 120,
    2000: 120,
    5000: 120,
}

# --------------------------------------------------------------------------
# 2. Argument Parser
# --------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="VRP Ablation Study Comparison Framework")
    # --- Test Configuration ---
    parser.add_argument('--problems', nargs='+', default=["CVRP"], 
                        help=f"List of VRP problem types to test, or 'ALL' to run all supported types: {SUPPORTED_PROBLEM_TYPES}")
    parser.add_argument('--sizes', nargs='+', type=int, default=PROBLEM_SIZES_TO_TEST, help="List of problem sizes (N) to test.")
    parser.add_argument('--num_instances', type=int, default=50, help="Number of instances to test per problem size/type.")
    parser.add_argument('--seed', type=int, default=2024, help="Random seed.")

    # --- Methods to Run ---
    parser.add_argument('--methods', type=str, nargs='+', default=ALL_METHODS, 
                        help="Method(s) or method group(s) to include in the ablation study. Available groups: AllSolver, AllORTools, AllPartitionSolver, AllPartitionORTools, AllPartition, All_m1.")
    # --- ADD test_batch_size ---
    parser.add_argument('--test_batch_size', type=int, default=32, 
                        help="Batch size for processing instances during testing for batched methods.")

    # --- Model Paths (Required for specific methods) ---
    parser.add_argument('--solver_checkpoint', type=str, default=None, 
                        help="Path to the pre-trained Solver model checkpoint. Required for *Solver methods.")
    parser.add_argument('--partitioner_checkpoint', type=str, default=None, 
                        help="Path to the pre-trained Partitioner model checkpoint. Required for Partition* methods.")

    # --- Solver Model Parameters --- 
    parser.add_argument('--solver_model_type', type=str, default="MOE", help="Solver model architecture type.")
    parser.add_argument('--solver_num_experts', type=int, default=4, help="Number of experts for Solver model (if MOE).")
    parser.add_argument('--solver_embedding_dim', type=int, default=128, help="(Optional) Embedding dimension for Solver model.")
    parser.add_argument('--solver_ff_hidden_dim', type=int, default=512, help="(Optional) Feed-forward hidden dim for Solver model.")
    parser.add_argument('--solver_encoder_layer_num', type=int, default=6, help="(Optional) Number of encoder layers for Solver model.")

    # --- Partitioner Model Parameters --- 
    parser.add_argument('--partitioner_model_type', type=str, default="MOE", help="Partitioner model architecture type.")
    parser.add_argument('--partitioner_num_experts', type=int, default=4, help="Number of experts for Partitioner model (if MOE).")
    parser.add_argument('--partitioner_embedding_dim', type=int, default=128, help="(Optional) Embedding dimension for Partitioner model.")
    parser.add_argument('--partitioner_ff_hidden_dim', type=int, default=512, help="(Optional) Feed-forward hidden dim for Partitioner model.")
    parser.add_argument('--partitioner_encoder_layer_num', type=int, default=6, help="(Optional) Number of encoder layers for Partitioner model.")
    # --- REMOVED partitioner_merge_num and adaptive_merge_target_size ---
    # parser.add_argument('--partitioner_merge_num', ...) 
    # parser.add_argument('--adaptive_merge_target_size', ...)

    # --- Method Specific Parameters ---
    parser.add_argument('--solver_aug_factor', type=int, default=1, help="Augmentation factor for Solver model inference.")
    # New arguments for OR-Tools stagnation control
    parser.add_argument('--ortools_stagnation_duration', type=int, default=10, 
                        help="OR-Tools: Duration (seconds) of no significant improvement to trigger stagnation stop. Default: 10s.")
    parser.add_argument('--ortools_min_improvement_pct', type=float, default=0.5, 
                        help="OR-Tools: Minimum percentage improvement required to reset stagnation timer. Default: 0.5 (for 0.5%%).")

    # --- Execution Control ---
    parser.add_argument('--workers', type=int, default=8, help="Max workers for parallel OR-Tools subproblem solving (instance-level for PartitionORTools and DirectORTools).")
    parser.add_argument('--subproblem_workers', type=int, default=4, 
                        help="Max workers for OR-Tools when solving subproblems within PartitionORTools methods. Default: 4. Set to 0 for os.cpu_count().")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use if CUDA is enabled.")

    # --- Output ---
    parser.add_argument('--output_csv', type=str, default='ablation_results.csv', help="Path to save the CSV results.")
    parser.add_argument('--log_file', type=str, default='ablation_test.log', help="Path to save the log file.")
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
# 4. Method-Specific Helper Functions / Tasks
# --------------------------------------------------------------------------

def _ortools_task_unpacker(params_tuple):
    """ Helper function to unpack arguments for solve_subproblem_ortools_task. """
    # Expects tuple: (subproblem_instance_tuple, problem_type, effective_timelimit)
    subproblem_instance_tuple, problem_type, timelimit = params_tuple
    return solve_subproblem_ortools_task(subproblem_instance_tuple, problem_type, timelimit)

def solve_subproblem_ortools_task(subproblem_instance_tuple, problem_type, timelimit):
    """ Wrapper task for solving a single subproblem using OR-Tools in a process pool. """
    try:
        # Lazily import OR-Tools within the task if using ProcessPoolExecutor
        # to avoid potential issues with pickling complex objects if imported globally.
        # However, if ortools_solve_vrp itself handles imports cleanly, this might not be needed.
        # Let's assume ortools_solve_vrp is importable or defined elsewhere for now.
        # from ortools_solver import ortools_solve_vrp # Example if needed
        
        # --- Attempt to import ortools_solve_vrp --- 
        ortools_solve_vrp_func = None
        try:
            ortools_solver_module = importlib.import_module('ortools_solver')
            ortools_solve_vrp_func = getattr(ortools_solver_module, 'ortools_solve_vrp')
        except (ImportError, AttributeError) as import_err:
             # Cannot proceed without the solver function
             # Log or print error if possible, but difficult from pool worker
             # Return a failure state
             return (float('inf'), [], f"Import Error: {import_err}") 

        if ortools_solve_vrp_func:
            cost, flat_route = ortools_solve_vrp_func(subproblem_instance_tuple, problem_type, timelimit)
            return (cost, flat_route, None) # Return cost, path, and None for error
        else:
            # Should have been caught by import check
            return (float('inf'), [], "OR-Tools solve function not found")
            
    except Exception as e:
        # Log exception if possible, return failure state
        # import traceback
        # error_str = traceback.format_exc()
        return (float('inf'), [], f"Error in OR-Tools task: {e}")

def run_batched_partition_ortools(
    batch_instance_tuples,
    problem_type,
    partitioner_model,
    partitioner_params,
    merge_num, # Explicit merge num passed based on method
    adaptive_target, # Explicit target passed based on method
    device,
    args, # Pass args for OR-Tools timelimit and workers
    timelimit # Pass calculated timelimit
):
    batch_size = len(batch_instance_tuples)
    if batch_size == 0: return []

    # --- Timing --- 
    batch_partition_time_start = time.time()

    # --- Partitioning (Instance-by-instance for now) ---
    per_instance_raw_sequences_and_subs = []
    for inst_tuple in batch_instance_tuples:
        _sub_tuples_inst, _raw_seq_inst = partition_instance(
            original_instance_tuple=inst_tuple,
            problem_type=problem_type,
            partitioner_checkpoint_path=None, 
            merge_num=merge_num, # Use passed merge_num
            device=device,
            partitioner_model_params=partitioner_params, 
            partitioner_model=partitioner_model,
            target_node_count_for_merge=adaptive_target # Use passed target
        )
        per_instance_raw_sequences_and_subs.append(
            (_raw_seq_inst if _raw_seq_inst else [], _sub_tuples_inst if _sub_tuples_inst else [])
        )
    batch_partition_time = time.time() - batch_partition_time_start

    # --- Subproblem Collection --- 
    batch_solving_time_start = time.time()
    all_subproblems_global = []
    instance_subproblem_metadata = [] # Stores (num_subproblems_for_this_orig_inst, start_idx_in_global)
    subproblem_orig_indices = [] # Store original instance index for each subproblem
    subproblem_tuples_for_pool = [] # Store (tuple, type, limit) for map

    for i in range(batch_size):
        _raw_seq, sub_tuples_for_instance = per_instance_raw_sequences_and_subs[i]
        num_subs_this_instance = len(sub_tuples_for_instance)
        start_idx_global = len(all_subproblems_global)
        instance_subproblem_metadata.append((num_subs_this_instance, start_idx_global))
        
        if num_subs_this_instance > 0:
            all_subproblems_global.extend(sub_tuples_for_instance)
            subproblem_orig_indices.extend([i] * num_subs_this_instance)
            for sub_tuple in sub_tuples_for_instance:
                # Pass the calculated effective timelimit for subproblems
                subproblem_tuples_for_pool.append((sub_tuple, problem_type, timelimit))

    # --- Parallel OR-Tools Solving --- 
    flat_ortools_results = [] # List to store (cost, path, error_msg) tuples
    if subproblem_tuples_for_pool:
        # Use the new subproblem_workers argument for inner parallelism
        num_sub_workers = args.subproblem_workers if args.subproblem_workers > 0 else os.cpu_count()
        num_sub_workers = min(num_sub_workers, len(subproblem_tuples_for_pool)) # Don't use more workers than tasks
        logging.info(f"Solving {len(subproblem_tuples_for_pool)} subproblems using OR-Tools with {num_sub_workers} subproblem_workers...")
        
        # Use ThreadPoolExecutor if ortools_solve_vrp is thread-safe (often is) or ProcessPoolExecutor otherwise
        # ProcessPoolExecutor is generally safer for potentially complex/stateful tasks
        # executor_class = concurrent.futures.ThreadPoolExecutor
        executor_class = concurrent.futures.ProcessPoolExecutor
        
        try:
            with executor_class(max_workers=num_sub_workers) as executor:
                # Use map to preserve order implicitly matching subproblem_tuples_for_pool
                # func = partial(solve_subproblem_ortools_task, problem_type=problem_type, timelimit=args.ortools_timelimit)
                # The task wrapper now handles the args
                results_iterator = executor.map(_ortools_task_unpacker, subproblem_tuples_for_pool)
                flat_ortools_results = list(results_iterator) # Collect results
        except Exception as pool_exec_error:
            logging.error(f"Error during parallel OR-Tools execution (subproblems): {pool_exec_error}", exc_info=True)
            # Populate results with failures if pool crashed
            flat_ortools_results = [(float('inf'), [], f"Subproblem Pool Error: {pool_exec_error}")] * len(subproblem_tuples_for_pool)

        # Basic check on results length
        if len(flat_ortools_results) != len(subproblem_tuples_for_pool):
            logging.error("Parallel OR-Tools subproblem result count mismatch!")
            # Handle mismatch - maybe fill remaining with errors?
            needed = len(subproblem_tuples_for_pool) - len(flat_ortools_results)
            flat_ortools_results.extend([(float('inf'), [], "Subproblem Result Count Mismatch")] * needed)

    batch_solving_time = time.time() - batch_solving_time_start

    # --- Aggregate results per original instance --- 
    final_results_for_batch = []
    for i in range(batch_size):
        num_subproblems_this_orig_inst, start_idx_global_this_orig_inst = instance_subproblem_metadata[i]
        score_for_orig_instance = float('inf')

        if num_subproblems_this_orig_inst == 0:
            score_for_orig_instance = float('inf')
        elif not flat_ortools_results: # If pool failed or no subproblems
             score_for_orig_instance = float('inf')
        else:
            current_orig_inst_total_cost = 0
            sub_results_for_this_orig_inst = flat_ortools_results[start_idx_global_this_orig_inst : start_idx_global_this_orig_inst + num_subproblems_this_orig_inst]

            if len(sub_results_for_this_orig_inst) != num_subproblems_this_orig_inst:
                logging.error(f"Logic error: OR-Tools Sub-result slice for instance {i} incorrect.")
                current_orig_inst_total_cost = float('inf')
            else:
                for sub_cost, _sub_path, error_msg in sub_results_for_this_orig_inst:
                    if error_msg:
                         logging.warning(f"OR-Tools subproblem solve error for instance {i}: {error_msg}")
                    if sub_cost == float('inf') or sub_cost is None or math.isnan(sub_cost):
                        current_orig_inst_total_cost = float('inf')
                        logging.warning(f"Instance {i} failed due to OR-Tools subproblem failure.")
                        break
                    current_orig_inst_total_cost += sub_cost
            score_for_orig_instance = current_orig_inst_total_cost
        
        avg_inst_partition_time = batch_partition_time / batch_size if batch_size > 0 else 0
        avg_inst_solving_time = batch_solving_time / batch_size if batch_size > 0 else 0 # Crude average for parallel time

        final_results_for_batch.append({
            'score': score_for_orig_instance,
            'total_time_seconds': avg_inst_partition_time + avg_inst_solving_time,
            'partition_time_seconds': avg_inst_partition_time,
            'solving_time_seconds': avg_inst_solving_time,
            'num_subproblems': num_subproblems_this_orig_inst
        })

    return final_results_for_batch

def run_direct_solver_batched(
    batch_instance_tuples,
    problem_type,
    solver_model,
    solver_env_class,
    device,
    args
):
    """ Solves a batch of instances directly using the Solver model. """
    batch_size = len(batch_instance_tuples)
    if batch_size == 0: return []

    start_time = time.time()
    final_results_for_batch = []

    try:
        # Pad the original instances - Use the actual size of the instances in the batch
        # as the padding target, since we are solving them directly.
        # Find max N in this batch first.
        max_nodes_in_batch = 0
        key_to_index = {key: idx for idx, key in enumerate(VRP_DATA_FORMAT.get(problem_type, []))}
        node_xy_idx = key_to_index.get('node_xy')
        if node_xy_idx is None: raise ValueError("Cannot find 'node_xy' index")
        for inst_tuple in batch_instance_tuples:
             max_nodes_in_batch = max(max_nodes_in_batch, len(inst_tuple[node_xy_idx]))
        
        target_pad_size = max(1, max_nodes_in_batch) # Ensure at least 1

        padded_batch_tuples, _ = pad_subproblem_batch(batch_instance_tuples, problem_type, target_pad_size)
        if not padded_batch_tuples or len(padded_batch_tuples) != batch_size:
             raise ValueError("Failed to pad batch for direct solver.")

        # Prepare tensor data
        padded_batch_tensor_data = prepare_batch_tensor_data(
            padded_batch_tuples, problem_type, device
        )
        if not padded_batch_tensor_data:
             raise ValueError("Failed to prepare tensor data for direct solver.")

        # Solve using the solver model
        solver_model.eval() # Ensure eval mode
        solver_results = solve_vrp_batch(
            solver_model=solver_model,
            solver_env_class=solver_env_class,
            original_instance_tuples=batch_instance_tuples, # Pass original for pomo calc etc.
            padded_batch_data=padded_batch_tensor_data,
            padded_problem_size=target_pad_size, 
            problem_type=problem_type,
            device=device,
            aug_factor=args.solver_aug_factor
        )

        if not solver_results or len(solver_results) != batch_size:
            raise ValueError("Direct solver results count mismatch.")
            
        solve_time = time.time() - start_time
        avg_time_per_instance = solve_time / batch_size if batch_size > 0 else 0
        
        for i in range(batch_size):
            score, _path = solver_results[i]
            final_results_for_batch.append({
                'score': score if score != float('inf') and score is not None and not math.isnan(score) else float('inf'),
                'total_time_seconds': avg_time_per_instance,
                'partition_time_seconds': 0, # No partitioning
                'solving_time_seconds': avg_time_per_instance,
                'num_subproblems': 0 # No subproblems
            }) 
            
    except Exception as e:
        logging.error(f"Error in run_direct_solver_batched: {e}", exc_info=True)
        # Return failures for the entire batch
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / batch_size if batch_size > 0 else 0
        final_results_for_batch = [{
            'score': float('inf'), 'total_time_seconds': avg_time, 
            'partition_time_seconds': 0, 'solving_time_seconds': avg_time, 'num_subproblems': 0
        }] * batch_size

    return final_results_for_batch

def run_direct_ortools_instance(
    instance_tuple,
    problem_type,
    timelimit, # Use specific timelimit
    args_namespace # Pass args for stagnation control
):
    """ Solves a single instance directly using OR-Tools. """
    start_time = time.time()
    
    ortools_solve_vrp_func = None
    try:
        ortools_solver_module = importlib.import_module('ortools_solver')
        ortools_solve_vrp_func = getattr(ortools_solver_module, 'ortools_solve_vrp')
    except (ImportError, AttributeError) as import_err:
        logging.error(f"Cannot run DirectORTools: {import_err}")
        cost, path = float('inf'), []
    
    if ortools_solve_vrp_func:
        try:
            cost, path = ortools_solve_vrp_func(
                instance_tuple, 
                problem_type, 
                timelimit, # This is the effective_ortools_timelimit for the original problem
                stagnation_duration=args_namespace.ortools_stagnation_duration, # Pass from args
                min_stagnation_improvement_pct=args_namespace.ortools_min_improvement_pct # Pass from args
            )
        except Exception as e:
            logging.error(f"Error running OR-Tools directly: {e}", exc_info=True)
            cost, path = float('inf'), []
    else:
        # Error logged during import attempt
        cost, path = float('inf'), []
        
    solve_time = time.time() - start_time

    return {
        'score': cost if cost != float('inf') and cost is not None and not math.isnan(cost) else float('inf'),
        'total_time_seconds': solve_time,
        'partition_time_seconds': 0,
        'solving_time_seconds': solve_time,
        'num_subproblems': 0
    }

# Function for Partition+Solver (Batched)
def run_batched_partition_solve(
    batch_instance_tuples,
    problem_type,
    partitioner_model,
    partitioner_params,
    merge_num, # Explicit merge num based on method
    adaptive_target, # Explicit target based on method
    solver_model,
    solver_env_class,
    device,
    args
):
    """ Processes a batch for Partition+Solver methods. """
    batch_size = len(batch_instance_tuples)
    if batch_size == 0: return []

    # --- Timing ---
    batch_partition_time_start = time.time()

    # --- Partitioning (Instance-by-instance within batch) ---
    per_instance_raw_sequences_and_subs = []
    for inst_tuple in batch_instance_tuples:
        # Call partition_instance with the *explicit* merge config for this method
        _sub_tuples_inst, _raw_seq_inst = partition_instance(
            original_instance_tuple=inst_tuple,
            problem_type=problem_type,
            partitioner_checkpoint_path=None,
            merge_num=merge_num, # Use passed merge_num
            device=device,
            partitioner_model_params=partitioner_params,
            partitioner_model=partitioner_model,
            target_node_count_for_merge=adaptive_target # Use passed target (0 for dynamic)
        )
        per_instance_raw_sequences_and_subs.append(
            (_raw_seq_inst if _raw_seq_inst else [], _sub_tuples_inst if _sub_tuples_inst else [])
        )
    batch_partition_time = time.time() - batch_partition_time_start

    # --- Subproblem Collection --- 
    batch_solving_time_start = time.time()
    all_subproblems_global = []
    instance_subproblem_metadata = [] # Stores (num_subs, start_idx_global)

    for i in range(batch_size):
        _raw_seq, sub_tuples_for_instance = per_instance_raw_sequences_and_subs[i]
        num_subs_this_instance = len(sub_tuples_for_instance)
        start_idx_global = len(all_subproblems_global)
        instance_subproblem_metadata.append((num_subs_this_instance, start_idx_global))
        if num_subs_this_instance > 0:
            all_subproblems_global.extend(sub_tuples_for_instance)

    # --- Batched Solving of ALL collected subproblems --- 
    flat_solver_results = None
    if all_subproblems_global:
        try:
            # Determine padding size based on collected subproblems
            max_nodes_in_subs = 0
            key_to_index_sub = {key: idx for idx, key in enumerate(VRP_DATA_FORMAT.get(problem_type, []))}
            node_xy_idx_sub = key_to_index_sub.get('node_xy')
            if node_xy_idx_sub is None: raise ValueError("Cannot find 'node_xy' index for subproblems")
            for sub_tuple in all_subproblems_global:
                max_nodes_in_subs = max(max_nodes_in_subs, len(sub_tuple[node_xy_idx_sub]))
            target_pad_size_global = max(1, max_nodes_in_subs)
            
            padded_global_subproblems, _ = pad_subproblem_batch(all_subproblems_global, problem_type, target_pad_size_global)
            if not padded_global_subproblems: raise ValueError("Failed to pad global subproblem batch")
            tensor_data_global = prepare_batch_tensor_data(padded_global_subproblems, problem_type, device)
            if not tensor_data_global: raise ValueError("Failed to prepare tensor data for global subproblem batch")
            
            solver_model.eval()
            flat_solver_results = solve_vrp_batch(
                solver_model=solver_model,
                solver_env_class=solver_env_class,
                original_instance_tuples=all_subproblems_global,
                padded_batch_data=tensor_data_global,
                padded_problem_size=target_pad_size_global,
                problem_type=problem_type,
                device=device,
                aug_factor=args.solver_aug_factor
            )
            if not flat_solver_results or len(flat_solver_results) != len(all_subproblems_global):
                logging.error(f"Solver subproblem results mismatch. Expected {len(all_subproblems_global)}, got {len(flat_solver_results) if flat_solver_results else 0}.")
                flat_solver_results = None
        except Exception as e:
            logging.error(f"Error solving subproblem batch with Solver: {e}", exc_info=True)
            flat_solver_results = None
    batch_solving_time = time.time() - batch_solving_time_start

    # --- Aggregate results per original instance --- 
    final_results_for_batch = []
    for i in range(batch_size):
        num_subproblems_this_orig_inst, start_idx_global_this_orig_inst = instance_subproblem_metadata[i]
        score_for_orig_instance = float('inf')

        if num_subproblems_this_orig_inst == 0:
            score_for_orig_instance = float('inf')
        elif flat_solver_results is None:
            score_for_orig_instance = float('inf')
        else:
            current_orig_inst_total_cost = 0
            sub_results_for_this_orig_inst = flat_solver_results[start_idx_global_this_orig_inst : start_idx_global_this_orig_inst + num_subproblems_this_orig_inst]
            if len(sub_results_for_this_orig_inst) != num_subproblems_this_orig_inst:
                logging.error(f"Logic error: Solver sub-result slice for instance {i} incorrect.")
                current_orig_inst_total_cost = float('inf')
            else:
                for cost, _ in sub_results_for_this_orig_inst:
                    if cost == float('inf') or cost is None or math.isnan(cost):
                        current_orig_inst_total_cost = float('inf')
                        logging.warning(f"Instance {i} failed due to Solver subproblem failure.")
                        break
                    current_orig_inst_total_cost += cost
            score_for_orig_instance = current_orig_inst_total_cost
        
        avg_inst_partition_time = batch_partition_time / batch_size if batch_size > 0 else 0
        avg_inst_solving_time = batch_solving_time / batch_size if batch_size > 0 else 0

        final_results_for_batch.append({
            'score': score_for_orig_instance,
            'total_time_seconds': avg_inst_partition_time + avg_inst_solving_time,
            'partition_time_seconds': avg_inst_partition_time,
            'solving_time_seconds': avg_inst_solving_time,
            'num_subproblems': num_subproblems_this_orig_inst
        })
        
    return final_results_for_batch

# Helper function for batched DirectORTools execution
def _direct_ortools_task_runner(instance_tuple_and_problem_type_and_timelimit_and_args):
    """ Helper function to unpack arguments for run_direct_ortools_instance for ProcessPoolExecutor. """
    instance_tuple, problem_type, timelimit, args_namespace = instance_tuple_and_problem_type_and_timelimit_and_args
    return run_direct_ortools_instance(instance_tuple, problem_type, timelimit, args_namespace)

# Helper function for batched PartitionORTools execution (for a single original instance)
def _partition_ortools_task_runner(params_tuple):
    """ 
    Helper function to unpack arguments for run_batched_partition_ortools when processing 
    a single original instance in parallel.
    Expected params_tuple: 
        (instance_tuple, problem_type, partitioner_checkpoint_path, partitioner_params_dict, 
         merge_num, adaptive_target, device_for_load, args_namespace, effective_ortools_timelimit)
    """
    (instance_tuple, problem_type, p_checkpoint_path, p_params_dict, 
     merge_num, adaptive_target, p_device_obj, p_args_obj, p_timelimit) = params_tuple
    
    # --- Load partitioner model within the worker process ---
    # Attempt to import load_moe_model dynamically
    # This assumes partitioner_solver_utils.py is in a location accessible by Python's import system.
    # If issues persist, ensure PYTHONPATH is set correctly or use more robust relative imports if applicable.
    load_moe_model_func = None
    try:
        # Try importing directly assuming it's in the same directory or accessible path
        # This might need to be adjusted based on actual file structure and how modules are typically imported in your project
        # e.g., from .partitioner_solver_utils import load_moe_model if in a package
        # For a script, direct import from a file in the same dir might be 'from filename import func'
        # For now, using importlib as a more general approach if 'partitioner_solver_utils' is a module
        partitioner_utils_module = importlib.import_module('partitioner_solver_utils')
        load_moe_model_func = getattr(partitioner_utils_module, 'load_moe_model')
    except (ImportError, AttributeError) as import_err:
        logging.error(f"_partition_ortools_task_runner: Failed to import/find load_moe_model from partitioner_solver_utils: {import_err}", exc_info=True)
        return {'score': float('inf'), 'total_time_seconds': 0, 'partition_time_seconds': 0, 'solving_time_seconds': 0, 'num_subproblems': 0, 'error': 'ModelLoadImportError'}

    if load_moe_model_func is None: # Should be caught above, but as a safeguard
        logging.error(f"_partition_ortools_task_runner: load_moe_model_func is None after import attempt.")
        return {'score': float('inf'), 'total_time_seconds': 0, 'partition_time_seconds': 0, 'solving_time_seconds': 0, 'num_subproblems': 0, 'error': 'ModelLoadFuncNone'}

    loaded_partitioner_model = load_moe_model_func(
        p_checkpoint_path,
        p_device_obj,
        model_type=p_params_dict.get('model_type'), # Get type from params dict
        model_params=p_params_dict
    )
    
    if not loaded_partitioner_model:
        logging.error(f"_partition_ortools_task_runner: Failed to load partitioner model. Checkpoint: {p_checkpoint_path}")
        return {'score': float('inf'), 'total_time_seconds': 0, 'partition_time_seconds': 0, 'solving_time_seconds': 0, 'num_subproblems': 0, 'error': 'ModelLoadFailed'}
    
    # run_batched_partition_ortools expects a list of instances
    # It will return a list containing a single result dictionary
    # TIMELIMIT here is for the *original* problem size, subproblem specific limits are handled inside run_batched_partition_ortools
    results_list = run_batched_partition_ortools(
        batch_instance_tuples=[instance_tuple], # Pass as a batch of one
        problem_type=problem_type,
        partitioner_model=loaded_partitioner_model, # Use the newly loaded model
        partitioner_params=p_params_dict,           # Pass the params dict
        merge_num=merge_num,
        adaptive_target=adaptive_target,
        device=p_device_obj, # Device for partition_instance internal logic if it were to load
        args=p_args_obj, # Pass full args down for stagnation params to be used by ortools_solve_vrp via solve_subproblem_ortools_task
        timelimit=p_timelimit
    )
    if results_list: # Should contain one item
        return results_list[0]
    else: # Handle error case if run_batched_partition_ortools returns empty
        logging.error(f"_partition_ortools_task_runner: run_batched_partition_ortools returned empty for an instance.")
        return {'score': float('inf'), 'total_time_seconds': 0, 'partition_time_seconds': 0, 'solving_time_seconds': 0, 'num_subproblems': 0, 'error': 'RunBatchedEmptyResult'}

# --------------------------------------------------------------------------
# 5. Main Execution Logic
# --------------------------------------------------------------------------
def main():
    args = parse_arguments()
    # --- Setup Output Paths --- 
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_id = f"AblationRun_{run_timestamp}"
    base_output_dir = os.path.join('results', 'compare', run_id)
    try:
        os.makedirs(base_output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory {base_output_dir}: {e}. Files will be saved in current directory.")
        base_output_dir = "." # Fallback to current directory
        
    # Construct full paths using the base directory and filenames from args
    full_log_path = os.path.join(base_output_dir, args.log_file) 
    full_csv_path = os.path.join(base_output_dir, args.output_csv)
    
    # --- Setup Logging with Full Path --- 
    setup_logging(full_log_path, args.verbose_log)
    seed_everything(args.seed)

    # --- Unconditional Console Start Message ---
    print(f"\n>>> Starting Ablation Run: {run_id} <<<")
    print(f"    Logs: {full_log_path}")      # Show full path
    print(f"    Results: {full_csv_path}")    # Show full path
    if not args.verbose_log: print(f"    Use --verbose_log for detailed console output.")
    print("---")

    # --- Process problem types for 'ALL' --- 
    if len(args.problems) == 1 and args.problems[0].upper() == 'ALL':
        problems_to_iterate = SUPPORTED_PROBLEM_TYPES
        logging.info(f"Running for ALL supported problem types: {problems_to_iterate}")
    else:
        problems_to_iterate = args.problems
        logging.info(f"Running for specified problem types: {problems_to_iterate}")

    # --- Setup Device ---
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu_id}' if use_cuda else 'cpu')
    logging.info(f"Using device: {device}")
    args.gpu_id = args.gpu_id if use_cuda else None

    # --- Determine Resource Availability --- 
    solver_model_available = args.solver_checkpoint is not None
    partitioner_model_available = args.partitioner_checkpoint is not None
    ortools_available = False
    ortools_solve_vrp = None # Keep function pointer if available
    try:
        ortools_solver_module = importlib.import_module('ortools_solver')
        ortools_solve_vrp = getattr(ortools_solver_module, 'ortools_solve_vrp')
        ortools_available = True
        logging.info("OR-Tools module found and loaded.")
    except (ImportError, AttributeError):
        logging.warning("OR-Tools module (ortools_solver.py) not found or function missing. *ORTools methods disabled.")

    # --- Filter Requested Methods --- 
    expanded_methods = set()
    requested_methods_from_cli = args.methods

    for req_method_or_group in requested_methods_from_cli:
        if req_method_or_group == 'solver':
            expanded_methods.update(ALL_SOLVER_METHODS)
        elif req_method_or_group == 'orools':
            expanded_methods.update(ALL_ORTOOLS_METHODS)
        elif req_method_or_group == 'partition_solver':
            expanded_methods.update(ALL_PARTITION_SOLVER_METHODS)
        elif req_method_or_group == 'partition_ortools':
            expanded_methods.update(ALL_PARTITION_ORTOOLS_METHODS)
        elif req_method_or_group == 'partition':
            expanded_methods.update(ALL_PARTITION_METHODS)
        elif req_method_or_group == 'm1':
            expanded_methods.update(ALL_M1_METHODS)
        elif req_method_or_group == 'm3':
            expanded_methods.update(ALL_M3_METHODS)
        elif req_method_or_group == 'adaptive':
            expanded_methods.update(ALL_ADAPTIVE_METHODS)
        elif req_method_or_group == 'ALL':
            expanded_methods.update(ALL_METHODS)
        elif req_method_or_group in ALL_METHODS:
            expanded_methods.add(req_method_or_group)
        else:
            logging.warning(f"Unknown method or group: {req_method_or_group}. It will be ignored.")

    final_requested_methods = list(expanded_methods)
    
    methods_to_run = []
    if 'DirectSolver' in final_requested_methods:
        if solver_model_available: methods_to_run.append('DirectSolver')
        else: logging.warning("Skipping DirectSolver (requires --solver_checkpoint).")
    if 'DirectORTools' in final_requested_methods:
        if ortools_available: methods_to_run.append('DirectORTools')
        else: logging.warning("Skipping DirectORTools (OR-Tools not found).")
    # Check Partition+Solver methods
    for m_name in ['PartitionSolver_m1', 'PartitionSolver_m3', 'PartitionSolver_adaptive']:
        if m_name in final_requested_methods:
            if partitioner_model_available and solver_model_available: methods_to_run.append(m_name)
            else: logging.warning(f"Skipping {m_name} (requires --partitioner_checkpoint AND --solver_checkpoint).")
    # Check Partition+OR-Tools methods
    for m_name in ['PartitionORTools_m1', 'PartitionORTools_m3', 'PartitionORTools_adaptive']:
        if m_name in final_requested_methods:
            if partitioner_model_available and ortools_available: methods_to_run.append(m_name)
            else: logging.warning(f"Skipping {m_name} (requires --partitioner_checkpoint AND OR-Tools).")

    if not methods_to_run:
        logging.error("No methods can be run based on provided arguments and available resources. Exiting.")
        print("Error: No methods to run. Check arguments and ensure models/OR-Tools are available.")
        return
    logging.info(f"Methods to run in this session: {methods_to_run}")

    # --- Load Models (Load ONCE) --- 
    solver_model = None
    solver_params = None
    partitioner_model = None
    partitioner_params = None
    all_env_classes = {} # Store loaded env classes {problem_type: EnvClass}

    if any('Solver' in m for m in methods_to_run):
        logging.info("Loading Solver model...")
        try:
            solver_params = DEFAULT_MODEL_PARAMS.copy()
            solver_params['model_type'] = args.solver_model_type
            solver_params['num_experts'] = args.solver_num_experts
            solver_params['device'] = device
            if args.solver_embedding_dim is not None: solver_params['embedding_dim'] = args.solver_embedding_dim
            if args.solver_ff_hidden_dim is not None: solver_params['ff_hidden_dim'] = args.solver_ff_hidden_dim
            if args.solver_encoder_layer_num is not None: solver_params['encoder_layer_num'] = args.solver_encoder_layer_num
            # Note: Problem type is set later when loading
            solver_model = load_moe_model(args.solver_checkpoint, device, model_type=args.solver_model_type, model_params=solver_params)
            if not solver_model: raise ValueError("Solver model loading failed.")
            solver_model.eval()
            logging.info(f"Solver model loaded from {args.solver_checkpoint}")
        except Exception as e:
            logging.error(f"Failed to load Solver model: {e}", exc_info=True)
            print(f"Error: Failed to load Solver model from {args.solver_checkpoint}. *Solver methods disabled.")
            # Disable solver methods if loading failed
            methods_to_run = [m for m in methods_to_run if 'Solver' not in m]
            solver_model_available = False # Update availability flag
    
    if any('Partition' in m for m in methods_to_run):
        logging.info("Loading Partitioner model...")
        try:
            partitioner_params = DEFAULT_MODEL_PARAMS.copy()
            partitioner_params['model_type'] = args.partitioner_model_type
            partitioner_params['num_experts'] = args.partitioner_num_experts
            partitioner_params['device'] = device
            if args.partitioner_embedding_dim is not None: partitioner_params['embedding_dim'] = args.partitioner_embedding_dim
            if args.partitioner_ff_hidden_dim is not None: partitioner_params['ff_hidden_dim'] = args.partitioner_ff_hidden_dim
            if args.partitioner_encoder_layer_num is not None: partitioner_params['encoder_layer_num'] = args.partitioner_encoder_layer_num
            # Note: Problem type is set later
            partitioner_model = load_moe_model(args.partitioner_checkpoint, device, model_type=args.partitioner_model_type, model_params=partitioner_params)
            if not partitioner_model: raise ValueError("Partitioner model loading failed.")
            partitioner_model.eval()
            logging.info(f"Partitioner model loaded from {args.partitioner_checkpoint}")
        except Exception as e:
            logging.error(f"Failed to load Partitioner model: {e}", exc_info=True)
            print(f"Error: Failed to load Partitioner model from {args.partitioner_checkpoint}. Partition* methods disabled.")
            methods_to_run = [m for m in methods_to_run if 'Partition' not in m]
            partitioner_model_available = False
            
    if not methods_to_run:
        logging.error("No methods remaining after model loading failures. Exiting.")
        print("Error: No methods remaining after model loading failures.")
        return
    logging.info(f"Final methods to run after loading checks: {methods_to_run}")

    # --- Prepare CSV Output --- 
    csv_headers = ['problem_type', 'problem_size', 'instance_index', 'method',
                   'score', 'total_time_seconds',
                   'partition_time_seconds', 'solving_time_seconds', 'num_subproblems']
                   # Removed 'current_adaptive_target' as method name implies config
    csvfile = None
    writer = None
    try:
        # --- Use Full Path for CSV --- 
        csvfile = open(full_csv_path, 'w', newline='')
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        logging.info(f"Opened {full_csv_path} for writing results.")
    except IOError as e:
        logging.error(f"Error opening CSV file {full_csv_path}: {e}. Exiting.")
        print(f"Error: Cannot open output file {full_csv_path}. Check permissions.")
        return

    # --- Main Test Loop --- 
    global_start_time = time.time()
    total_instances_processed = 0

    for problem_type in problems_to_iterate:
        logging.info(f"--- Testing Problem Type: {problem_type} ---")
        # Get Env Class once per problem type
        CurrentEnvClass = None
        if problem_type not in all_env_classes:
            try:
                EnvClassList = get_env(problem_type)
                if not EnvClassList: raise ValueError(f"No EnvClass found for {problem_type}")
                all_env_classes[problem_type] = EnvClassList[0]
                logging.info(f"Using Env Class: {all_env_classes[problem_type].__name__}")
            except Exception as e:
                logging.error(f"Failed to get Env class for {problem_type}: {e}. Skipping problem type.")
                continue
        CurrentEnvClass = all_env_classes[problem_type]

        for size in args.sizes:
            logging.info(f"--- Testing Size: {size} ---")
            # Load dataset
            full_dataset_for_size = None
            try:
                dataset_path = None
                if problem_type in DATASET_PATHS and size in DATASET_PATHS[problem_type]:
                    dataset_info = DATASET_PATHS[problem_type][size]
                    dataset_path = dataset_info['data']
                    if not os.path.exists(dataset_path):
                         script_dir = os.path.dirname(__file__) if "__file__" in locals() else "."
                         relative_path = os.path.join(script_dir, dataset_path)
                         if os.path.exists(relative_path): dataset_path = relative_path
                         else: dataset_path = None # Not found
                if dataset_path is None:
                     logging.warning(f"Dataset path definition not found or file missing for {problem_type} size {size}. Skipping size.")
                     continue
                     
                full_dataset_for_size = load_dataset(dataset_path)
                num_available = len(full_dataset_for_size)
                instances_to_run_count = min(args.num_instances, num_available)
                if instances_to_run_count < args.num_instances: logging.warning(f"Requested {args.num_instances}, only {num_available} available for N={size}.")
                if instances_to_run_count <= 0: logging.warning(f"No instances to run for N={size}. Skipping."); continue
                logging.info(f"Testing {instances_to_run_count} instances for N={size} from {dataset_path}")
                instances_to_process_all = full_dataset_for_size[:instances_to_run_count]
            except Exception as e:
                logging.error(f"Failed to load dataset for {problem_type} N={size}: {e}. Skipping size.")
                continue

            # Determine effective BATCH size for this problem size
            base_batch_size = args.test_batch_size
            cap_batch_size = base_batch_size 
            
            # Apply caps similar to node_num_target_comparison.py or train_partitioner.py
            if size >= 5000: cap_batch_size = max(1, base_batch_size // 4) # Stricter cap for very large
            elif size >= 2000: cap_batch_size = max(1, base_batch_size // 2) # cap to 4 if base is 32, or 1-2 if smaller base
            # elif size >= 1000: cap_batch_size = max(1, base_batch_size // 2) # Example: cap to 16
            # Add more elif conditions here for other sizes if needed
            
            effective_test_batch_size = min(base_batch_size, cap_batch_size)
            effective_test_batch_size = max(1, effective_test_batch_size) # Ensure batch size is at least 1
            
            if effective_test_batch_size < base_batch_size:
                logging.info(f"Testing N={size}: Dynamically capping test batch size from {base_batch_size} to {effective_test_batch_size}.")

            # --- Determine Effective OR-Tools Timelimit for this size using hardcoded map --- 
            predefined_sizes = np.array(list(ORTOOLS_TIMELIMIT_MAP.keys()))
            # Find the index of the closest predefined size
            closest_index = (np.abs(predefined_sizes - size)).argmin()
            closest_size = predefined_sizes[closest_index]
            effective_ortools_timelimit = ORTOOLS_TIMELIMIT_MAP[closest_size]
            logging.info(f"For N={size}, closest predefined size is {closest_size}. Using OR-Tools timelimit: {effective_ortools_timelimit}s")

            # --- Method Loop ---
            # Outer progress bar for methods
            pbar_methods = tqdm(methods_to_run, desc=f"Methods {problem_type}_N{size}", unit="method", leave=True, position=0)
            for method in pbar_methods:
                pbar_methods.set_description(f"Method: {method} ({problem_type} N={size})") # Update outer pbar description

                logging.info(f"--- Running Method: {method} for {problem_type} N={size} (Batch Size: {effective_test_batch_size}) ---")
                
                num_batches = math.ceil(instances_to_run_count / effective_test_batch_size)
                # Inner progress bar for batches of the current method
                pbar_batches = tqdm(range(num_batches), desc=f"{method} {problem_type}_N{size}", unit="batch", leave=False, position=1)

                for batch_idx in pbar_batches:
                    batch_start_idx = batch_idx * effective_test_batch_size
                    batch_end_idx = min(batch_start_idx + effective_test_batch_size, instances_to_run_count)
                    current_batch_instance_tuples = instances_to_process_all[batch_start_idx:batch_end_idx]
                    current_actual_batch_size = len(current_batch_instance_tuples)

                    if current_actual_batch_size == 0:
                        continue
                    
                    total_instances_processed += current_actual_batch_size # Update count here

                    logging.debug(f"Method '{method}', Batch {batch_idx+1}/{num_batches} (Size: {current_actual_batch_size}, Indices: {batch_start_idx}-{batch_end_idx-1})")
                    
                    method_results_for_batch = []
                    
                    try:
                        if method == 'DirectSolver':
                             if solver_model and CurrentEnvClass:
                                 method_results_for_batch = run_direct_solver_batched(
                                     current_batch_instance_tuples, problem_type, solver_model, 
                                     CurrentEnvClass, device, args
                                 )
                             else: 
                                 logging.error("Solver/Env not ready for DirectSolver. Returning failure for batch.")
                                 method_results_for_batch = [{'score': float('inf'), 'total_time_seconds': 0, 'partition_time_seconds': 0, 'solving_time_seconds': 0, 'num_subproblems': 0}] * current_actual_batch_size
                        
                        elif method == 'DirectORTools':
                            if ortools_available:
                                # Prepare tasks with (instance_tuple, problem_type, effective_timelimit, args)
                                tasks_for_direct_ortools = [
                                    (inst_tuple, problem_type, effective_ortools_timelimit, args) 
                                    for inst_tuple in current_batch_instance_tuples
                                ]
                                
                                num_workers_ortools = args.workers if args.workers > 0 else os.cpu_count()
                                num_workers_ortools = min(num_workers_ortools, current_actual_batch_size)

                                if num_workers_ortools > 1 and current_actual_batch_size > 1: # Only use pool for multiple tasks
                                    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers_ortools) as executor:
                                        # Ensure we are mapping over the 4-element task tuples
                                        method_results_for_batch = list(executor.map(_direct_ortools_task_runner, tasks_for_direct_ortools))
                                else: # Process sequentially
                                    method_results_for_batch = [
                                        run_direct_ortools_instance(inst_tuple, problem_type, effective_ortools_timelimit, args) 
                                        for inst_tuple in current_batch_instance_tuples
                                    ]
                            else: 
                                logging.error("OR-Tools not available for DirectORTools. Returning failure for batch.")
                                method_results_for_batch = [{'score': float('inf'), 'total_time_seconds': 0, 'partition_time_seconds': 0, 'solving_time_seconds': 0, 'num_subproblems': 0}] * current_actual_batch_size
                             
                        elif method.startswith('PartitionSolver'):
                            if partitioner_model and solver_model and CurrentEnvClass:
                                merge_num = -1 # Default adaptive
                                if method == 'PartitionSolver_m1': merge_num = 1
                                elif method == 'PartitionSolver_m3': merge_num = 3
                                method_results_for_batch = run_batched_partition_solve(
                                    current_batch_instance_tuples, problem_type, 
                                    partitioner_model, partitioner_params, 
                                    merge_num, 0, # adaptive_target (0 for dynamic if merge_num=-1)
                                    solver_model, CurrentEnvClass, 
                                    device, args
                                )
                            else: 
                                logging.error("Model/Env not ready for PartitionSolver method. Returning failure for batch.")
                                method_results_for_batch = [{'score': float('inf'), 'total_time_seconds': 0, 'partition_time_seconds': 0, 'solving_time_seconds': 0, 'num_subproblems': 0}] * current_actual_batch_size
                            
                        elif method.startswith('PartitionORTools'):
                            if partitioner_model and ortools_available:
                                merge_num = -1 # Default adaptive
                                if method == 'PartitionORTools_m1': merge_num = 1
                                elif method == 'PartitionORTools_m3': merge_num = 3
                                
                                # Prepare tasks for ProcessPoolExecutor, one task per original instance in the batch
                                partition_ortools_tasks = []
                                for inst_tuple in current_batch_instance_tuples:
                                    partition_ortools_tasks.append((
                                        inst_tuple, problem_type, 
                                        args.partitioner_checkpoint, # MODIFIED: Pass checkpoint path
                                        partitioner_params,          # MODIFIED: Pass model parameter dict
                                        merge_num, 0, # adaptive_target (0 for dynamic if merge_num=-1)
                                        device, args, effective_ortools_timelimit
                                    ))
                                
                                num_workers_part_ortools = args.workers if args.workers > 0 else os.cpu_count()
                                num_workers_part_ortools = min(num_workers_part_ortools, current_actual_batch_size)

                                if num_workers_part_ortools > 1 and current_actual_batch_size > 1:
                                    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers_part_ortools) as executor:
                                        method_results_for_batch = list(executor.map(_partition_ortools_task_runner, partition_ortools_tasks))
                                else: # Process sequentially if only one task or one worker
                                    temp_results = []
                                    for task_params in partition_ortools_tasks:
                                        temp_results.append(_partition_ortools_task_runner(task_params))
                                    method_results_for_batch = temp_results

                            else: 
                                logging.error("Model/OR-Tools not ready for PartitionORTools method. Returning failure for batch.")
                                method_results_for_batch = [{'score': float('inf'), 'total_time_seconds': 0, 'partition_time_seconds': 0, 'solving_time_seconds': 0, 'num_subproblems': 0}] * current_actual_batch_size
                        
                        # Ensure method_results_for_batch has correct length if a step failed internally
                        if len(method_results_for_batch) != current_actual_batch_size:
                            logging.error(f"Method {method} returned {len(method_results_for_batch)} results for batch of size {current_actual_batch_size}. Filling with errors.")
                            # Fill with error dicts if counts don't match
                            error_results = [{'score': float('inf'), 'total_time_seconds': 0, 'partition_time_seconds': 0, 'solving_time_seconds': 0, 'num_subproblems': 0}]
                            method_results_for_batch.extend(error_results * (current_actual_batch_size - len(method_results_for_batch)))
                            
                    except Exception as method_batch_err:
                        logging.error(f"Error running method {method} for batch {batch_idx}: {method_batch_err}", exc_info=True)
                        method_results_for_batch = [{'score': float('inf'), 'total_time_seconds': 0, 'partition_time_seconds': 0, 'solving_time_seconds': 0, 'num_subproblems': 0}] * current_actual_batch_size
                    
                    # Write results for each instance in the batch
                    for k, result_dict in enumerate(method_results_for_batch):
                        instance_index_in_dataset = batch_start_idx + k 
                    writer.writerow({
                        'problem_type': problem_type,
                        'problem_size': size,
                        'instance_index': instance_index_in_dataset,
                        'method': method,
                            'score': f"{result_dict.get('score', float('inf')):.4f}" if result_dict.get('score', float('inf')) != float('inf') else 'inf',
                            'total_time_seconds': f"{result_dict.get('total_time_seconds', 0.0):.4f}",
                            'partition_time_seconds': f"{result_dict.get('partition_time_seconds', 0.0):.4f}",
                            'solving_time_seconds': f"{result_dict.get('solving_time_seconds', 0.0):.4f}",
                            'num_subproblems': result_dict.get('num_subproblems', 0)
                    }) 
                    csvfile.flush() 
                    # Add memory cleanup after each batch, especially for large N
                    if use_cuda:
                        torch.cuda.empty_cache()
                    gc.collect()
                    logging.debug(f"Memory cleanup after batch {batch_idx} for method {method}, {problem_type} N={size}")
                # --- End Batch Loop ---
                pbar_batches.close() # Close inner progress bar

                # Memory cleanup after processing all batches for a method+size+problem_type
                # This one might be redundant if we clean after each batch, but can be kept as a final measure.
                if use_cuda: torch.cuda.empty_cache()
                gc.collect()
                logging.debug(f"Memory cleanup after method {method} for {problem_type} N={size}")
            
            pbar_methods.close() # Close outer progress bar after all methods for this size are done
            # --- End Method Loop --- 
            logging.info(f"--- Finished Size: {size} ---")
        # --- End Size Loop ---
        logging.info(f"--- Finished Problem Type: {problem_type} ---")
    # --- End Problem Loop ---

    global_end_time = time.time()
    logging.info(f"Ablation test finished. Processed {total_instances_processed} total instance-method evaluations.")
    logging.info(f"Total time: {global_end_time - global_start_time:.2f} seconds.")
    
    # --- Final Cleanup and Console Message --- 
    if csvfile:
        try: csvfile.close() 
        except: pass
        logging.info(f"Results file closed: {full_csv_path}") # Use full path
        
    print("---")
    print(f">>> Finished Ablation Run: {run_id} <<<" ) 
    print(f"    Results saved to: {full_csv_path}") # Use full path
    print(f"    Detailed logs in: {full_log_path}") # Use full path


# --------------------------------------------------------------------------
# 6. Script Execution Guard
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for CUDA compatibility with ProcessPoolExecutor
    # This needs to be done before any multiprocessing-related objects are created or CUDA is initialized in parent.
    import multiprocessing as mp
    # Check if the start method has already been set (e.g., by another library or in a notebook)
    # and only set it if it's not already 'spawn' or if it hasn't been set at all.
    # Force=True is used to ensure our setting takes precedence if it was set to something else by default.
    if mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
            print(f"INFO: Multiprocessing start method set to 'spawn'.")
        except RuntimeError as e:
            # This can happen if context has already been used (e.g. pool created before this point)
            # Or if CUDA was initialized in parent and fork is default, then spawn is attempted too late.
            # Best practice is to set this at the very beginning.
            print(f"WARNING: Could not set multiprocessing start method to 'spawn': {e}. " 
                  f"If CUDA errors persist, ensure this is at the script's entry point before other imports/ops.")

    main()
