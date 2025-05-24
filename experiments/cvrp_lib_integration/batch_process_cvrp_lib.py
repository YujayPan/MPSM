import os
import sys
import torch
import pickle
import json
import time
import numpy as np
import math
import argparse
import csv
import glob
from tqdm import tqdm
import logging
import importlib # For dynamically importing ortools_solver
import concurrent.futures
import multiprocessing as mp

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cvrp_lib_integration.cvrp_lib_parser import parse_vrp_file, to_internal_tuple
from utils import VRP_DATA_FORMAT, seed_everything, get_env
from partitioner_solver_utils import (
    load_moe_model, DEFAULT_MODEL_PARAMS,
    _split_sequence_by_zeros,
    merge_subproblems_by_centroid_fixed_size,
    create_subproblem_instance,
    pad_subproblem_batch, prepare_batch_tensor_data, solve_vrp_batch
)

# --- Constants for Methods (similar to solve_from_partitions.py) ---
ALL_METHODS_SUPPORTED = [
    'DirectSolver',
    'DirectORTools',
    'PartitionSolver_m1',
    'PartitionSolver_m3',
    'PartitionSolver_adaptive',
    'PartitionORTools_m1',
    'PartitionORTools_m3',
    'PartitionORTools_adaptive',
]

MERGE_CONFIG_TO_SUFFIX = {
    'PartitionSolver_m1': 'm1',
    'PartitionSolver_m3': 'm3',
    'PartitionSolver_adaptive': 'adaptive',
    'PartitionORTools_m1': 'm1',
    'PartitionORTools_m3': 'm3',
    'PartitionORTools_adaptive': 'adaptive',
}

# For OR-Tools timelimits (can be adjusted or made configurable)
ORTOOLS_TIMELIMIT_MAP = { 
    50: 2,
    100: 2,
    200: 5,
    500: 10,
    1000: 20,
    2000: 40,
    5000: 60,
}

# --- Helper functions ---
def parse_mc_string(config_name: str):
    """ Parses a merge configuration string (e.g., 'm1', 'adaptive', 'adaptive_s50'). """
    if config_name == 'raw_subroutes': # Should ideally be handled before calling this
        return None, None
    elif config_name == 'm1':
        return 1, 0  # merge_num=1, adaptive_target=0 (not used for fixed)
    elif config_name == 'm3':
        return 3, 0  # merge_num=3, adaptive_target=0
    elif config_name == 'adaptive':
        return -1, 0 # merge_num=-1 (triggers adaptive), adaptive_target=0 (dynamic selection)
    elif config_name.startswith('adaptive_s'):
        try:
            target_size = int(config_name.split('adaptive_s')[-1])
            return -1, target_size # merge_num=-1, explicit adaptive_target
        except ValueError:
            logging.warning(f"Could not parse adaptive target size from '{config_name}'. Using dynamic selection.")
            return -1, 0 # Fallback to dynamic if parsing fails
    else:
        logging.warning(f"Unknown merge configuration name for parsing: '{config_name}'.")
        return None, None # Unknown

def generate_sequence_and_initial_routes_from_tuple(
    original_instance_tuple, 
    problem_type, 
    loaded_partitioner_model,
    device_obj,
    max_seq_len_factor=2
):
    raw_sequence = None
    initial_subproblem_node_lists = []
    try:
        node_xy_index = VRP_DATA_FORMAT[problem_type].index('node_xy')
        num_customer_nodes = len(original_instance_tuple[node_xy_index])
        if num_customer_nodes <= 0:
            logging.error(f"(gen_seq): Instance for {problem_type} has no customer nodes.")
            return None, []
        
        max_seq_len = max_seq_len_factor * (num_customer_nodes + 1)
        EnvClassList = get_env(problem_type)
        if not EnvClassList:
            logging.error(f"(gen_seq): Could not get env class for {problem_type}")
            return None, []
        PartitionEnvClass = EnvClassList[0]
        env_params = {"problem_size": num_customer_nodes, "pomo_size": 1, "device": device_obj}
        partition_env = PartitionEnvClass(**env_params)
        
        padded_batch_tuples, target_pad_size = pad_subproblem_batch(
            [original_instance_tuple], problem_type, num_customer_nodes
        )
        if not padded_batch_tuples or target_pad_size != num_customer_nodes:
            logging.error(f"(gen_seq): Padding/Target size mismatch. Expected {num_customer_nodes}, got {target_pad_size}")
            return None, []
        
        instance_tensor_data = prepare_batch_tensor_data(padded_batch_tuples, problem_type, device_obj)
        if not instance_tensor_data:
            logging.error("(gen_seq): Failed to prepare instance tensor data for partitioning.")
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
                logging.warning(f"(gen_seq): Sequence generation reached max length ({max_seq_len}).")
            if hasattr(partition_env, 'selected_node_list') and partition_env.selected_node_list is not None:
                if partition_env.selected_node_list.numel() > 0:
                    raw_sequence = partition_env.selected_node_list.view(-1).cpu().tolist()
                else: logging.warning("(gen_seq): partition_env.selected_node_list is empty.")
            else: logging.warning("(gen_seq): partition_env.selected_node_list not found.")
        if raw_sequence is None: return None, []
        initial_subproblem_node_lists = _split_sequence_by_zeros(raw_sequence)
    except Exception as e:
        logging.error(f"Error in generate_sequence_and_initial_routes_from_tuple: {e}", exc_info=True)
        return None, []
    return raw_sequence, initial_subproblem_node_lists

def run_direct_nn_solver_instance(
    original_instance_tuple_for_processing, # Can be normalized or raw based on need
    problem_type,
    solver_model,
    device,
    solver_aug_factor,
    solver_model_params # Pass full solver_model_params dict
):
    """ Solves a single instance directly using the NN Solver. """
    solve_time_start = time.time()
    EnvClassList = get_env(problem_type)
    if not EnvClassList:
        return {'score': float('inf'), 'solve_time_seconds': 0, 'num_subproblems': 0, 'error': 'NoEnvClass_DirectNN', 'full_path_original_indices': []}
    SolverEnvClass = EnvClassList[0]

    try:
        node_xy_idx = VRP_DATA_FORMAT[problem_type].index('node_xy')
        num_nodes_in_instance = len(original_instance_tuple_for_processing[node_xy_idx])
        target_pad_size = max(1, num_nodes_in_instance)

        padded_batch_tuples, actual_pad_size = pad_subproblem_batch(
            [original_instance_tuple_for_processing], problem_type, target_pad_size
        )
        if not padded_batch_tuples or actual_pad_size != target_pad_size:
            return {'score': float('inf'), 'solve_time_seconds': time.time() - solve_time_start, 'num_subproblems': 0, 'error': 'DirectNN_PaddingError', 'full_path_original_indices': []}

        padded_tensor_data = prepare_batch_tensor_data(padded_batch_tuples, problem_type, device)
        if not padded_tensor_data:
            return {'score': float('inf'), 'solve_time_seconds': time.time() - solve_time_start, 'num_subproblems': 0, 'error': 'DirectNN_TensorPrepError', 'full_path_original_indices': []}

        solver_model.eval()
        results = solve_vrp_batch(
            solver_model=solver_model,
            solver_env_class=SolverEnvClass,
            original_instance_tuples=[original_instance_tuple_for_processing],
            padded_batch_data=padded_tensor_data,
            padded_problem_size=actual_pad_size,
            problem_type=problem_type,
            device=device,
            aug_factor=solver_aug_factor
        )
        if not results or len(results) != 1:
            return {'score': float('inf'), 'solve_time_seconds': time.time() - solve_time_start, 'num_subproblems': 0, 'error': 'DirectNN_SolveError', 'full_path_original_indices': []}
        
        score, path_nodes = results[0]
        # For direct solver, path_nodes are already 0-based relative to the instance *passed to it* (which is 1-indexed customer list)
        # Need to convert them to 1-based for consistency in path reconstruction if needed by cost calculator.
        # However, the cost calculator calculate_path_cost_original_coords expects 0 for depot, 1..N for customers.
        # solve_vrp_batch path is 0 for depot, 1..N for customers.
        reconstructed_path_for_cost_calc = path_nodes if path_nodes else []

    except Exception as e:
        logging.error(f"Error in run_direct_nn_solver_instance: {e}", exc_info=True)
        return {'score': float('inf'), 'solve_time_seconds': time.time() - solve_time_start, 'num_subproblems': 0, 'error': f'DirectNN_Exception: {e}', 'full_path_original_indices': []}
    
    current_solve_time = time.time() - solve_time_start
    return {
        'score': score if score is not None and not torch.isnan(torch.tensor(score)) else float('inf'),
        'solve_time_seconds': current_solve_time,
        'num_subproblems': 0, # Direct solving has 0 subproblems by definition
        'full_path_original_indices': reconstructed_path_for_cost_calc,
        'error': None
    }

def run_partition_nn_solver_instance(
    original_instance_tuple_for_processing, # Normalized instance
    problem_type, 
    subproblem_node_lists_from_merge, # 1-based indices relative to original_instance_tuple_for_processing
    solver_model, 
    device, 
    solver_aug_factor,
    solver_model_params # Pass full solver_model_params dict
):
    if not original_instance_tuple_for_processing or not solver_model or not subproblem_node_lists_from_merge:
        return {'score': float('inf'), 'solve_time_seconds': 0, 'num_subproblems': 0, 'error': 'MissingInput_PartNN', 'full_path_original_indices': []}
    solve_time_start = time.time()
    EnvClassList = get_env(problem_type)
    if not EnvClassList:
        return {'score': float('inf'), 'solve_time_seconds': 0, 'num_subproblems': 0, 'error': 'NoEnvClass_PartNN', 'full_path_original_indices': []}
    SolverEnvClass = EnvClassList[0]
    subproblem_instance_tuples = [] # These will be normalized if original_instance_tuple_for_processing is
    num_original_nodes_in_processed_instance = 0
    try:
        node_xy_idx_orig = VRP_DATA_FORMAT[problem_type].index('node_xy')
        num_original_nodes_in_processed_instance = len(original_instance_tuple_for_processing[node_xy_idx_orig]) 
    except Exception as e:
        return {'score': float('inf'), 'solve_time_seconds': 0, 'num_subproblems': 0, 'error': f'OriginalNodeAccessError_PartNN: {e}', 'full_path_original_indices': []}

    for original_indices_for_current_sub in subproblem_node_lists_from_merge:
        if not original_indices_for_current_sub: continue
        # These indices are 1-based from the original problem, map to the current (potentially normalized) original_instance_tuple_for_processing
        valid_indices_for_sub = [idx for idx in original_indices_for_current_sub if 1 <= idx <= num_original_nodes_in_processed_instance]
        if not valid_indices_for_sub: continue
        sub_instance = create_subproblem_instance(original_instance_tuple_for_processing, problem_type, valid_indices_for_sub)
        if sub_instance: subproblem_instance_tuples.append(sub_instance)
        else: logging.warning(f"(run_solver): Failed to create subproblem for {valid_indices_for_sub}")
    if not subproblem_instance_tuples:
        return {'score': float('inf'), 'solve_time_seconds': time.time() - solve_time_start, 'num_subproblems': 0, 'error': 'NoValidSubproblemsCreated_PartNN', 'full_path_original_indices': []}
    
    num_actual_subproblems = len(subproblem_instance_tuples)
    max_nodes_in_sub_batch = 0
    try: node_xy_idx_sub = VRP_DATA_FORMAT[problem_type].index('node_xy')
    except Exception: return {'score': float('inf'), 'solve_time_seconds': time.time() - solve_time_start, 'num_subproblems': num_actual_subproblems, 'error': 'SubproblemNodeXYIndexError_PartNN', 'full_path_original_indices': []}
    
    for sub_inst_tuple in subproblem_instance_tuples:
        try: max_nodes_in_sub_batch = max(max_nodes_in_sub_batch, len(sub_inst_tuple[node_xy_idx_sub]))
        except Exception: pass 
            
    target_pad_size_for_subs = max(1, max_nodes_in_sub_batch)
    padded_subproblem_batch, actual_pad_size_for_subs = pad_subproblem_batch(
        subproblem_instance_tuples, problem_type, target_pad_size_for_subs
    )
    if not padded_subproblem_batch or len(padded_subproblem_batch) != num_actual_subproblems:
        return {'score': float('inf'), 'solve_time_seconds': time.time() - solve_time_start, 'num_subproblems': num_actual_subproblems, 'error': 'SubproblemPaddingError_PartNN', 'full_path_original_indices': []}
    padded_subproblem_tensor_data = prepare_batch_tensor_data(padded_subproblem_batch, problem_type, device)
    if not padded_subproblem_tensor_data:
        return {'score': float('inf'), 'solve_time_seconds': time.time() - solve_time_start, 'num_subproblems': num_actual_subproblems, 'error': 'SubproblemTensorPrepError_PartNN', 'full_path_original_indices': []}
    solver_model.eval()
    flat_solver_results = [] # List of (score, path_list)
    # Solve subproblems in batches if too many for memory (optional enhancement, for now solve all at once)
    try:
        flat_solver_results = solve_vrp_batch(
            solver_model=solver_model,
            solver_env_class=SolverEnvClass,
            original_instance_tuples=subproblem_instance_tuples, # Pass the unpadded subproblem tuples
            padded_batch_data=padded_subproblem_tensor_data,
            padded_problem_size=actual_pad_size_for_subs,
            problem_type=problem_type,
            device=device,
            aug_factor=solver_aug_factor
        )
    except Exception as e:
        logging.error(f"Exception during solve_vrp_batch for subproblems: {e}", exc_info=True)
        # Populate with failures to match expected length if solve_vrp_batch crashes
        flat_solver_results = [(float('inf'), None)] * num_actual_subproblems

    # Add detailed logging for flat_solver_results
    logging.debug(f"PartitionNN: flat_solver_results (first 5): {flat_solver_results[:5]}")
    if not flat_solver_results or len(flat_solver_results) != num_actual_subproblems:
        logging.error(f"PartitionNN: Subproblem solver results count mismatch. Expected {num_actual_subproblems}, got {len(flat_solver_results) if flat_solver_results else 0}.")
        # Populate with failures if mismatch to prevent crash, ensure error is logged
        error_fill_count = num_actual_subproblems - (len(flat_solver_results) if flat_solver_results else 0)
        flat_solver_results = (flat_solver_results if flat_solver_results else []) + [(float('inf'), None)] * error_fill_count
        # Ensure the error propogates
        method_execution_result['error'] = "SubproblemSolveCountMismatch_PartNN"

    # 4. Aggregate scores and Reconstruct Full Path with ORIGINAL 1-based indices
    total_aggregated_score_normalized = 0
    full_path_original_indices = []
    all_subproblems_solved_successfully = True
    # Initialize a dictionary within this function to store results including potential errors
    method_execution_result = {
        'normalized_score': float('inf'),
        'full_path_original_indices': [],
        'error': None
    }

    # Log the subproblem_node_lists_from_merge to understand the mapping
    logging.debug(f"PartitionNN: subproblem_node_lists_from_merge (first 5): {subproblem_node_lists_from_merge[:5]}")

    for i, (sub_score_norm, sub_path_indices_relative_to_sub) in enumerate(flat_solver_results):
        logging.debug(f"PartitionNN: Processing subproblem {i}: Score={sub_score_norm}, RelativePath={sub_path_indices_relative_to_sub}")
        if sub_score_norm == float('inf') or sub_score_norm is None or math.isnan(sub_score_norm) or sub_path_indices_relative_to_sub is None:
            total_aggregated_score_normalized = float('inf')
            all_subproblems_solved_successfully = False
            logging.warning(f"PartitionNN: Subproblem {i} failed to solve (score: {sub_score_norm}, path: {sub_path_indices_relative_to_sub}). Original instance score will be inf.")
            # If a subproblem fails, we cannot reliably reconstruct the full path for original cost calculation.
            # We should indicate this failure.
            if not method_execution_result['error']: # Don't overwrite a previous error like SubproblemSolveCountMismatch
                method_execution_result['error'] = "SubproblemSolveFailed_PartNN"
            # Break here as path reconstruction is no longer meaningful if one subproblem fails
            break 
        
        total_aggregated_score_normalized += sub_score_norm

        # Map sub_path_indices_relative_to_sub (0-based for subproblem, or 1-based if solver gives 1-based directly)
        # to original 1-based indices.
        # The subproblem_node_lists_from_merge contains the ORIGINAL 1-based indices for each subproblem.
        current_subproblem_original_nodes = subproblem_node_lists_from_merge[i] # This is a list of original 1-based indices

        if not full_path_original_indices: # First subproblem path
            pass # Will be added directly
        elif full_path_original_indices[-1] != 0: # If previous path didn't end at depot
            full_path_original_indices.append(0) # Add depot visit

        segment_added = False
        for node_idx_in_sub_path in sub_path_indices_relative_to_sub:
            if node_idx_in_sub_path == 0: # Depot visit within the subproblem's own path
                # If last in full_path was already 0, don't add another.
                if not full_path_original_indices or full_path_original_indices[-1] != 0:
                    full_path_original_indices.append(0)
            else:
                # Assuming sub_path_indices_relative_to_sub are 1-based relative to nodes IN THAT SUBPROBLEM
                # So, node_idx_in_sub_path = 1 means the first node in current_subproblem_original_nodes
                if 1 <= node_idx_in_sub_path <= len(current_subproblem_original_nodes):
                    # This is a real customer node within the subproblem's original span
                    original_node_idx = current_subproblem_original_nodes[node_idx_in_sub_path - 1]
                    full_path_original_indices.append(original_node_idx)
                    segment_added = True
                elif node_idx_in_sub_path > len(current_subproblem_original_nodes):
                    # This means the solver selected a PADDING node.
                    # This is not an error from the solver's perspective, as it operates on the padded instance.
                    # We simply ignore this padding node in the final reconstructed path.
                    logging.debug(f"PartitionNN: Subproblem {i}, solver selected a padding node (index {node_idx_in_sub_path} > original subproblem size {len(current_subproblem_original_nodes)}). Ignoring.")
                else:
                    # This case (e.g., node_idx_in_sub_path is < 1 but not 0) should ideally not happen if solver behaves.
                    logging.error(f"PartitionNN: Path reconstruction error for subproblem {i}! Invalid node index {node_idx_in_sub_path} in SubPath: {sub_path_indices_relative_to_sub}")
                    total_aggregated_score_normalized = float('inf') # Mark failure
                    all_subproblems_solved_successfully = False
                    if not method_execution_result['error']: method_execution_result['error'] = "PathReconstructionInvalidIndex_PartNN"
                    break # Stop processing this subproblem's path
        
        if not all_subproblems_solved_successfully: # If an error occurred in the inner loop
            break # Stop processing further subproblems for this instance

        # If a segment was added and it didn't naturally end with a depot connection,
        # and it's not the last subproblem, we might need a depot connection.
        # However, the main loop structure (adding 0 if previous didn't end with 0) should handle this.

    if not all_subproblems_solved_successfully and not method_execution_result['error']:
        # This case can happen if loop finishes but all_subproblems_solved_successfully is false
        # due to an earlier break from a subproblem solve failure, but no specific path error occurred in THIS loop.
        # The error "SubproblemSolveFailed_PartNN" should have been set. If not, set a general one.
        method_execution_result['error'] = "PathReconstructionFailed_PartNN" # Generic if specific error wasn't set

    # Final check: if the path ends with 0 (depot) and it's not the only element, remove it,
    # as the overall path is usually represented without a final return to depot IF it's an open problem type,
    # or if the cost function implicitly handles it. For CVRP, it should typically end at depot,
    # but our `calculate_path_cost_original_coords` adds the return trip.
    # For now, let's keep it simple: if it ends with 0, and is not just [0], pop it.
    # This matches the behavior of the original single_instance script.
    if full_path_original_indices and full_path_original_indices[-1] == 0 and len(full_path_original_indices) > 1:
        # Check if the second to last is also 0 (e.g. ...0,0 from a subproblem ending at depot)
        # If so, this might be okay. But if it's like [1,2,3,0], it's a return to depot.
        # The original single script did this:
        # if solution_path and solution_path[-1] == 0: solution_path.pop()
        # Let's replicate that for now.
        # full_path_original_indices.pop()
        pass # Let's not pop for now, calculate_path_cost_original_coords handles return to depot.

    method_execution_result['normalized_score'] = round(total_aggregated_score_normalized, 4) if total_aggregated_score_normalized != float('inf') else 'inf'
    method_execution_result['full_path_original_indices'] = full_path_original_indices if all_subproblems_solved_successfully else [] # Return empty path on error

    solve_time_seconds = round(time.time() - solve_time_start, 4)
    method_execution_result['time_solve_nn_sec'] = solve_time_seconds

    return method_execution_result

def calculate_path_cost_original_coords(flat_path_original_indices, depot_xy_raw_list, nodes_xy_raw_list):
    if not flat_path_original_indices or len(flat_path_original_indices) < 2: return 0.0
    # Ensure depot_xy_raw_list is a list of lists, e.g., [[x, y]]
    depot_coord = depot_xy_raw_list[0] if depot_xy_raw_list and isinstance(depot_xy_raw_list[0], list) else depot_xy_raw_list
    all_coords_raw = [depot_coord] + nodes_xy_raw_list
    total_distance = 0.0
    for i in range(len(flat_path_original_indices) - 1):
        from_node_original_idx = flat_path_original_indices[i]
        to_node_original_idx = flat_path_original_indices[i+1]
        try:
            coord_from = all_coords_raw[from_node_original_idx] 
            coord_to = all_coords_raw[to_node_original_idx]
        except IndexError: return float('inf')
        total_distance += math.sqrt((coord_from[0] - coord_to[0])**2 + (coord_from[1] - coord_to[1])**2)
    return total_distance

# --- OR-Tools Helper Function Placeholder ---
def run_direct_ortools_instance(original_instance_tuple_raw, problem_type, timelimit, ortools_solve_func):
    # This will be implemented in the next step
    logging.info(f"Running DirectORTools for {problem_type} with timelimit {timelimit}s.")
    if not ortools_solve_func:
        return {'score': float('inf'), 'solve_time_seconds': 0, 'num_subproblems': 0, 'error': 'ORToolsFuncNotProvided_Direct', 'full_path_original_indices': []}
    solve_time_start = time.time()
    cost, path = ortools_solve_func(original_instance_tuple_raw, problem_type, timelimit)
    solve_time_s = time.time() - solve_time_start
    return {
        'score': cost if cost is not None else float('inf'), # OR-Tools cost is already original
        'solve_time_seconds': solve_time_s,
        'num_subproblems': 0,
        'full_path_original_indices': path if path else [],
        'error': None if cost != float('inf') else 'ORToolsDirectFail'
    }

# Helper for ProcessPoolExecutor for PartitionORTools in this script
def _ortools_subproblem_task_runner_batch(params_tuple):
    """
    Unpacks arguments and calls the OR-Tools solver for a single subproblem.
    Expected params_tuple: (subproblem_instance_tuple, problem_type, 
                              timelimit_for_subproblem, ortools_solve_vrp_function_to_call,
                              stagnation_args_tuple (duration, min_improvement_pct))
    Returns: (cost, path_list) tuple from ortools_solve_vrp_function_to_call
    """
    sub_instance, prob_type, sub_timelimit, ortools_func, stagnation_args = params_tuple
    stagnation_duration, min_stagnation_improvement_pct = stagnation_args

    if not ortools_func:
        logging.error("_ortools_subproblem_task_runner_batch: OR-Tools function not provided.")
        return float('inf'), []
    try:
        # Call the OR-Tools solver for the subproblem
        # Pass stagnation parameters to the solver function
        cost, path = ortools_func(
            sub_instance, 
            prob_type, 
            sub_timelimit,
            stagnation_duration=stagnation_duration, # from new arg
            min_stagnation_improvement_pct=min_stagnation_improvement_pct # from new arg
        )
        return cost, path
    except Exception as e:
        logging.error(f"Error in OR-Tools subproblem task (_ortools_subproblem_task_runner_batch): {e}", exc_info=True)
        return float('inf'), [] # Return failure state

def run_partition_ortools_instance(
    original_instance_tuple_raw, 
    problem_type, 
    subproblem_node_lists_from_merge, # 1-based indices relative to original_instance_tuple_raw
    timelimit_per_subproblem, 
    ortools_solve_func, 
    num_subproblem_workers=4,
    ortools_stagnation_duration: int = 10, # New: stagnation duration
    ortools_min_improvement_pct: float = 0.5 # New: min improvement for stagnation
    ):
    """
    Solves an original instance by creating subproblems from loaded node lists, 
    then solving these subproblems in parallel using OR-Tools.
    Returns a dictionary including the aggregated score (original coordinates) and total wall-clock solve time.
    """
    method_execution_result = {
        'score': float('inf'), # This will be cost
        'solve_time_seconds': 0,
        'num_subproblems': 0, # Will be updated
        'full_path_original_indices': [], # Will be reconstructed
        'error': None
    }
    
    solve_time_start = time.time()

    if not original_instance_tuple_raw or not ortools_solve_func:
        logging.error("run_partition_ortools_instance: Missing original instance or OR-Tools function.")
        method_execution_result['error'] = "MissingInputOrORToolsFunc_PartOR"
        method_execution_result['solve_time_seconds'] = time.time() - solve_time_start
        return method_execution_result

    if not subproblem_node_lists_from_merge:
        logging.warning(f"run_partition_ortools_instance: Received empty subproblem_node_lists for {problem_type}. Returning inf score.")
        method_execution_result['error'] = "EmptyNodeLists_PartOR"
        method_execution_result['solve_time_seconds'] = time.time() - solve_time_start
        return method_execution_result

    # 1. Create subproblem instance tuples (using original raw coordinates)
    subproblem_instance_tuples = []
    num_original_nodes = 0
    try: 
        # Assuming VRP_DATA_FORMAT is accessible or we infer from a standard CVRP-like structure
        # For CVRP-LIB, the tuple structure is ("depot_xy", "node_xy", "demand", "capacity", ...)
        # depot_xy_raw_list = original_instance_tuple_raw[0]
        nodes_xy_raw_list = original_instance_tuple_raw[1] 
        num_original_nodes = len(nodes_xy_raw_list)
    except (IndexError, TypeError) as e:
        logging.error(f"run_partition_ortools_instance: Error accessing raw node_xy for count: {e}", exc_info=True)
        method_execution_result['error'] = "OriginalNodeAccessError_PartOR"
        method_execution_result['solve_time_seconds'] = time.time() - solve_time_start
        return method_execution_result

    for node_indices_1based_list in subproblem_node_lists_from_merge:
        if not node_indices_1based_list: continue
        valid_indices_for_sub = [idx for idx in node_indices_1based_list if 1 <= idx <= num_original_nodes]
        if not valid_indices_for_sub: continue
        
        # create_subproblem_instance expects VRP_DATA_FORMAT, which might not be directly available here
        # or match the CVRP-LIB specific tuple. Let's adapt its core logic.
        # The input `original_instance_tuple_raw` has format:
        # ("depot_xy", "node_xy", "demand", "capacity", "problem_type", "dimension", "name")
        depot_xy_sub = original_instance_tuple_raw[0]
        all_nodes_xy_sub = original_instance_tuple_raw[1]
        all_demands_sub = original_instance_tuple_raw[2]
        capacity_sub = original_instance_tuple_raw[3]
        # problem_type_sub = original_instance_tuple_raw[4] # This is the CVRP-LIB type, not necessarily what OR-Tools func expects for format.
                                                        # The passed 'problem_type' arg to this function should guide OR-Tools format.

        sub_node_xy = [all_nodes_xy_sub[i-1] for i in valid_indices_for_sub] # 0-indexed from 1-based
        sub_demand = [all_demands_sub[i-1] for i in valid_indices_for_sub]

        # Construct the subproblem tuple in the format expected by `ortools_solve_func`
        # This relies on `ortools_solve_func` using `parse_vrp_instance_tuple` which uses `VRP_DATA_FORMAT`.
        # We need to ensure the `problem_type` argument passed to this function correctly reflects that format.
        
        # For now, assume problem_type is 'CVRP' or similar for OR-Tools parsing.
        # If other VRP types from VRP_DATA_FORMAT need to be supported here, this construction needs to be more robust.
        if problem_type == "CVRP": # Example: Construct for CVRP type
            sub_instance = (depot_xy_sub, sub_node_xy, sub_demand, capacity_sub)
        else:
            # This part needs to be generalized if `ortools_solve_func` is to handle various problem types
            # based on the `VRP_DATA_FORMAT` structure used by `parse_vrp_instance_tuple`.
            # For now, we log an error if it's not a simple CVRP for this OR-Tools path.
            # A more robust solution would be to use `create_subproblem_instance` from `partitioner_solver_utils`
            # if VRP_DATA_FORMAT and the original tuple structure align with its expectations.
            # Given `original_instance_tuple_raw` is from CVRP-LIB parser, direct construction is safer if `problem_type` is fixed.
            logging.error(f"run_partition_ortools_instance: problem_type '{problem_type}' subproblem construction not fully implemented beyond CVRP. Adapt if needed.")
            # Fallback to CVRP structure for now, or error out
            sub_instance = (depot_xy_sub, sub_node_xy, sub_demand, capacity_sub) # Assuming CVRP structure
            # method_execution_result['error'] = f"UnsupportedProblemTypeForSubproblemCreation_PartOR:{problem_type}"
            # method_execution_result['solve_time_seconds'] = time.time() - solve_time_start
            # return method_execution_result


        if sub_instance: subproblem_instance_tuples.append(sub_instance)

    if not subproblem_instance_tuples:
        logging.warning(f"No valid subproblem instances created from loaded node lists for {problem_type} (PartitionORTools). Score inf.")
        method_execution_result['error'] = "NoValidSubproblemsCreated_PartOR"
        method_execution_result['num_subproblems'] = 0
        method_execution_result['solve_time_seconds'] = time.time() - solve_time_start
        return method_execution_result
    
    num_actual_subproblems = len(subproblem_instance_tuples)
    method_execution_result['num_subproblems'] = num_actual_subproblems

    # 2. Parallel OR-Tools Solving of subproblems
    subproblem_tasks_for_pool = []
    stagnation_params_for_task = (ortools_stagnation_duration, ortools_min_improvement_pct)
    for sub_inst_tuple in subproblem_instance_tuples:
        subproblem_tasks_for_pool.append((
            sub_inst_tuple, 
            problem_type, # Use the problem_type passed to this function, assumed to be what ortools_solve_func expects
            timelimit_per_subproblem,
            ortools_solve_func,
            stagnation_params_for_task # Pass tuple of stagnation params
        ))

    flat_ortools_results = [] # List to store (cost, path) tuples from subproblem solutions
    
    actual_num_workers = num_subproblem_workers
    if actual_num_workers <= 0 : # Use os.cpu_count() if 0 or negative
        actual_num_workers = os.cpu_count() or 1 # Default to 1 if os.cpu_count() is None
    # Cap workers by number of tasks and available CPUs
    actual_num_workers = min(actual_num_workers, num_actual_subproblems, os.cpu_count() or 1) 
    
    logging.info(f"Solving {num_actual_subproblems} OR-Tools subproblems for {problem_type} using {actual_num_workers} workers (requested: {num_subproblem_workers}). Timelimit/sub: {timelimit_per_subproblem}s")
    
    if subproblem_tasks_for_pool:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=actual_num_workers) as executor:
                # Pass the new helper function that accepts stagnation params
                results_iterator = executor.map(_ortools_subproblem_task_runner_batch, subproblem_tasks_for_pool)
                flat_ortools_results = list(results_iterator) 
        except Exception as pool_exec_error:
            logging.error(f"Error during parallel OR-Tools execution (subproblems): {pool_exec_error}", exc_info=True)
            flat_ortools_results = [(float('inf'), [])] * num_actual_subproblems # Fill with failures
            method_execution_result['error'] = "ParallelExecutionError_PartOR"

        if len(flat_ortools_results) != num_actual_subproblems:
            logging.error("Parallel OR-Tools subproblem result count mismatch!")
            needed = num_actual_subproblems - len(flat_ortools_results)
            flat_ortools_results.extend([(float('inf'), [])] * needed)
            if not method_execution_result['error']: # Don't overwrite a previous error
                 method_execution_result['error'] = "ResultCountMismatch_PartOR"
    
    # 3. Aggregate scores and Reconstruct Full Path
    total_aggregated_score_original_coords = 0
    full_path_original_indices = [] 
    all_subproblems_solved_successfully = True

    for i, (sub_cost, sub_path_indices_relative_to_sub) in enumerate(flat_ortools_results):
        if sub_cost == float('inf') or sub_cost is None or math.isnan(sub_cost) or sub_path_indices_relative_to_sub is None:
            total_aggregated_score_original_coords = float('inf')
            all_subproblems_solved_successfully = False
            logging.warning(f"PartitionORTools: Subproblem {i} failed to solve (cost: {sub_cost}). Original instance score will be inf.")
            if not method_execution_result['error']:
                method_execution_result['error'] = "SubproblemSolveFailed_PartOR"
            break # Path reconstruction is not meaningful if one subproblem fails
        
        total_aggregated_score_original_coords += sub_cost

        current_subproblem_original_nodes = subproblem_node_lists_from_merge[i]

        if not full_path_original_indices: pass
        elif full_path_original_indices[-1] != 0:
            full_path_original_indices.append(0)

        for node_idx_in_sub_path in sub_path_indices_relative_to_sub: # These are 1-based from OR-Tools solution for the subproblem
            if node_idx_in_sub_path == 0:
                if not full_path_original_indices or full_path_original_indices[-1] != 0:
                    full_path_original_indices.append(0)
            else: # Customer node from subproblem path
                # OR-Tools paths are 1-based for customers *within that subproblem instance*.
                # The subproblem instance itself was created using nodes from `current_subproblem_original_nodes`.
                # So, node_idx_in_sub_path = 1 means the first node in the subproblem's own node list.
                # This corresponds to current_subproblem_original_nodes[0] if we 0-indexed current_subproblem_original_nodes.
                # Since current_subproblem_original_nodes contains original 1-based indices:
                try:
                    # Check if node_idx_in_sub_path is within the bounds of how many nodes were in THIS subproblem
                    # The nodes in sub_path_indices_relative_to_sub are 1...k where k is num customers in that subproblem.
                    # This k should be len(current_subproblem_original_nodes) for this subproblem 'i'.
                    if 1 <= node_idx_in_sub_path <= len(current_subproblem_original_nodes):
                         original_node_idx = current_subproblem_original_nodes[node_idx_in_sub_path - 1]
                         full_path_original_indices.append(original_node_idx)
                    else:
                        # This indicates an issue with the path returned by OR-Tools for the subproblem,
                        # or a misunderstanding of its indexing.
                        logging.error(f"PartitionORTools: Path reconstruction error for subproblem {i}! Node index {node_idx_in_sub_path} from OR-Tools path is out of bounds for the original nodes in this subproblem (len {len(current_subproblem_original_nodes)}). SubPath from OR-Tools: {sub_path_indices_relative_to_sub}")
                        total_aggregated_score_original_coords = float('inf')
                        all_subproblems_solved_successfully = False
                        if not method_execution_result['error']: method_execution_result['error'] = "PathRecIndexErrorORTools_PartOR"
                        break 
                except IndexError: # Should be caught by the check above, but as a safeguard
                    logging.error(f"PartitionORTools: Unexpected IndexError during path reconstruction for subproblem {i}.")
                    total_aggregated_score_original_coords = float('inf')
                    all_subproblems_solved_successfully = False
                    if not method_execution_result['error']: method_execution_result['error'] = "PathRecUnexpectedIndexErrorORTools_PartOR"
                    break
        
        if not all_subproblems_solved_successfully: break
    
    if not all_subproblems_solved_successfully and not method_execution_result['error'] :
         method_execution_result['error'] = "PathReconstructionFailed_PartOR"


    method_execution_result['score'] = round(total_aggregated_score_original_coords, 5) if total_aggregated_score_original_coords != float('inf') else 'inf'
    method_execution_result['full_path_original_indices'] = full_path_original_indices if all_subproblems_solved_successfully else []
    method_execution_result['solve_time_seconds'] = round(time.time() - solve_time_start, 4)
    
    return method_execution_result

# --- End Helper Functions ---

DEFAULT_MERGE_CONFIGS_STR = ['adaptive', 'm1', 'm3'] # Used by NN-Solver methods
# ORTools partition methods also use these merge configs for their partitions

def parse_arguments():
    parser = argparse.ArgumentParser(description="Batch Processing for CVRP-LIB Instances with Multiple Methods")
    parser.add_argument('--cvrp_lib_dir', type=str, default=os.path.join(project_root, "CVRP-LIB"),
                        help="Directory containing .vrp files.")
    parser.add_argument('--output_root_dir', type=str, default=os.path.join(project_root, "cvrp_lib_integration", "run_outputs_batch_multi_method"),
                        help="Root directory to save all batch processing results.")
    # Partitioner model args
    parser.add_argument('--partitioner_checkpoint', type=str, default=None,
                        help="Path to the Partitioner model checkpoint (fine-tuned Solver). Required for partition-based methods.")
    parser.add_argument('--partitioner_model_type', type=str, default="MOE")
    parser.add_argument('--partitioner_num_experts', type=int, default=4)
    parser.add_argument('--partitioner_embedding_dim', type=int, default=128)
    parser.add_argument('--partitioner_ff_hidden_dim', type=int, default=512)
    parser.add_argument('--partitioner_encoder_layer_num', type=int, default=6)
    # Solver model args (for NN Solver methods)
    parser.add_argument('--solver_checkpoint', type=str, default=None,
                        help="Path to the Solver model checkpoint. Required for *Solver (NN-based) methods.")
    parser.add_argument('--solver_model_type', type=str, default="MOE")
    parser.add_argument('--solver_num_experts', type=int, default=8)
    parser.add_argument('--solver_embedding_dim', type=int, default=128)
    parser.add_argument('--solver_ff_hidden_dim', type=int, default=512)
    parser.add_argument('--solver_encoder_layer_num', type=int, default=6)
    parser.add_argument('--solver_aug_factor', type=int, default=1, help="Augmentation for NN Solver.")
    # OR-Tools specific args
    parser.add_argument('--ortools_subproblem_workers', type=int, default=4, help="Number of workers for OR-Tools subproblem solving (0 for os.cpu_count()).")
    parser.add_argument('--ortools_stagnation_duration', type=int, default=10, help="OR-Tools: Stagnation duration (seconds) for subproblem solver. Default: 10s.")
    parser.add_argument('--ortools_min_improvement_pct', type=float, default=0.5, help="OR-Tools: Min improvement percentage for stagnation in subproblem solver. Default: 0.5 (0.5%%).")
    # Method selection
    parser.add_argument('--methods', nargs='+', type=str, default=['ALL_METHODS_SUPPORTED'], # Default to run all, can be overridden
                        help=f"List of methods to run. Subsets like 'Direct*', 'PartitionSolver*', 'PartitionORTools*', or individual methods from {ALL_METHODS_SUPPORTED}. Default runs all.")
    parser.add_argument('--seed', type=int, default=2024, help="Random seed.")
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use for NN models.")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA for NN models.')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--verbose_console', action='store_true', help="Enable verbose logging to console (INFO level). Otherwise, console shows WARNING and above.")

    args = parser.parse_args()
    return args

def setup_logging(log_file_path, level_str, verbose_console):
    numeric_level = getattr(logging, level_str.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level_str}')
    
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s')
    
    root_logger = logging.getLogger() 
    root_logger.setLevel(min(logging.DEBUG, numeric_level)) # Root logger set to the finest level needed by any handler
    root_logger.handlers.clear() 
    
    # File Handler - always logs from the level specified by --log_level
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(numeric_level) # File handler respects --log_level
    root_logger.addHandler(file_handler)
    
    # Console Handler - logs WARNING and above, or INFO if verbose_console is true
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    if verbose_console:
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.WARNING)
    root_logger.addHandler(console_handler)

def get_effective_ortools_timelimit(problem_size_n, timelimit_map, default_timelimit):
    predefined_ort_sizes = np.array(sorted(list(timelimit_map.keys())))
    if not predefined_ort_sizes.size: return default_timelimit
    effective_tl = timelimit_map.get(predefined_ort_sizes[-1], default_timelimit)
    if problem_size_n <= predefined_ort_sizes[0]:
        effective_tl = timelimit_map.get(predefined_ort_sizes[0], default_timelimit)
    else:
        for i in range(len(predefined_ort_sizes) - 1):
            if problem_size_n > predefined_ort_sizes[i] and problem_size_n <= predefined_ort_sizes[i+1]:
                effective_tl = timelimit_map.get(predefined_ort_sizes[i+1], default_timelimit)
                break
    return effective_tl

def main_batch_processor():
    args = parse_arguments()
    seed_everything(args.seed)
    use_cuda_nn = not args.no_cuda and torch.cuda.is_available()
    device_str_nn = f'cuda:{args.gpu_id}' if use_cuda_nn else 'cpu'
    device_nn = torch.device(device_str_nn)

    batch_run_id = f"batch_run_mm_{time.strftime('%Y%m%d_%H%M%S')}"
    current_run_output_dir = os.path.join(args.output_root_dir, batch_run_id); os.makedirs(current_run_output_dir, exist_ok=True)
    log_file_full_path = os.path.join(current_run_output_dir, f"batch_processing_{batch_run_id}.log")
    setup_logging(log_file_full_path, args.log_level, args.verbose_console)

    logging.info(f"Starting Batch CVRP-LIB Multi-Method Processing. Run ID: {batch_run_id}")
    logging.info(f"Using device for NN models: {device_str_nn}")
    logging.info(f"Output for this batch run will be in: {current_run_output_dir}")
    logging.info(f"Script arguments: {args}")

    # --- Determine Methods to Run ---
    requested_methods_processed = set()
    if 'ALL_METHODS_SUPPORTED' in args.methods or 'ALL' in args.methods:
        requested_methods_processed.update(ALL_METHODS_SUPPORTED)
    else:
        for method_or_group in args.methods:
            if method_or_group == 'DirectSolver': requested_methods_processed.add('DirectSolver')
            elif method_or_group == 'DirectORTools': requested_methods_processed.add('DirectORTools')
            elif method_or_group.startswith('PartitionSolver'): # e.g. PartitionSolver_m1, PartitionSolver_all
                if method_or_group == 'PartitionSolver_all':
                    requested_methods_processed.update([m for m in ALL_METHODS_SUPPORTED if m.startswith('PartitionSolver')])
                elif method_or_group in ALL_METHODS_SUPPORTED: requested_methods_processed.add(method_or_group)
            elif method_or_group.startswith('PartitionORTools'):
                if method_or_group == 'PartitionORTools_all':
                    requested_methods_processed.update([m for m in ALL_METHODS_SUPPORTED if m.startswith('PartitionORTools')])
                elif method_or_group in ALL_METHODS_SUPPORTED: requested_methods_processed.add(method_or_group)
            elif method_or_group in ALL_METHODS_SUPPORTED: requested_methods_processed.add(method_or_group)
            else: logging.warning(f"Unknown method or group in --methods: {method_or_group}. It will be ignored.")
    
    methods_to_run_final = list(requested_methods_processed)
    logging.info(f"Methods to be run in this session: {methods_to_run_final}")

    # --- Load Resources Conditionally ---
    partitioner_model, solver_model_nn, ortools_solve_vrp_func = None, None, None
    if any(m.startswith('Partition') for m in methods_to_run_final):
        if not args.partitioner_checkpoint: logging.critical("Partitioner checkpoint needed for partition-based methods but not provided. Exiting."); return
        logging.info(f"Loading Partitioner Model ({args.partitioner_model_type}) from {args.partitioner_checkpoint}")
        part_params = DEFAULT_MODEL_PARAMS.copy()
        for k in ['model_type', 'num_experts', 'embedding_dim', 'ff_hidden_dim', 'encoder_layer_num']: part_params[k] = getattr(args, f'partitioner_{k}')
        part_params['device'] = device_nn
        partitioner_model = load_moe_model(args.partitioner_checkpoint, device_nn, model_type=args.partitioner_model_type, model_params=part_params)
        if not partitioner_model: logging.critical("Failed to load partitioner model. Exiting."); return
        logging.info("Partitioner model loaded.")

    if any('Solver' in m for m in methods_to_run_final): # NN Solver methods
        if not args.solver_checkpoint: logging.critical("Solver checkpoint needed for NN-Solver methods but not provided. Exiting."); return
        logging.info(f"Loading NN Solver Model ({args.solver_model_type}) from {args.solver_checkpoint}")
        sol_params = DEFAULT_MODEL_PARAMS.copy()
        for k in ['model_type', 'num_experts', 'embedding_dim', 'ff_hidden_dim', 'encoder_layer_num']: sol_params[k] = getattr(args, f'solver_{k}')
        sol_params['device'] = device_nn
        solver_model_nn = load_moe_model(args.solver_checkpoint, device_nn, model_type=args.solver_model_type, model_params=sol_params)
        if not solver_model_nn: logging.critical("Failed to load NN Solver model. Exiting."); return
        logging.info("NN Solver model loaded.")
    
    if any('ORTools' in m for m in methods_to_run_final):
        try:
            ortools_module = importlib.import_module("ortools_solver")
            ortools_solve_vrp_func = getattr(ortools_module, 'ortools_solve_vrp')
            logging.info("OR-Tools ortools_solve_vrp function loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load OR-Tools solver: {e}. ORTools methods will be skipped.", exc_info=True)
            methods_to_run_final = [m for m in methods_to_run_final if 'ORTools' not in m]

    if not methods_to_run_final: logging.critical("No runnable methods after checking resources. Exiting."); return
    logging.info(f"Final methods to execute after resource check: {methods_to_run_final}")

    csv_file_path = os.path.join(current_run_output_dir, "summary_results_multimethod.csv")
    csv_headers = [
        'problem_name', 'problem_size_N', 'method_applied', 
        'cost', 
        'time_parse_norm_sec', 'time_partition_gen_sec', 'time_merge_sec',
        'time_solve_sec', 'time_total_sec', 'num_subproblems'
    ]
    all_results_for_csv = []
    vrp_files = glob.glob(os.path.join(args.cvrp_lib_dir, "*.vrp"))
    if not vrp_files: logging.error(f"No .vrp files found in {args.cvrp_lib_dir}. Exiting."); return
    logging.info(f"Found {len(vrp_files)} .vrp files to process with {len(methods_to_run_final)} methods each.")

    for vrp_file_path in tqdm(vrp_files, desc="Processing VRP Files"):
        instance_basename = os.path.basename(vrp_file_path)
        instance_name_for_output = os.path.splitext(instance_basename)[0]
        logging.info(f"== Processing Instance: {instance_basename} ==")
        
        time_parse_start = time.time()
        parsed_data = parse_vrp_file(vrp_file_path)
        if not parsed_data: logging.error(f"Failed to parse {instance_basename}. Skipping."); continue
        original_instance_tuple_RAW = to_internal_tuple(parsed_data)
        if not original_instance_tuple_RAW: logging.error(f"Failed to convert {instance_basename} to internal tuple. Skipping."); continue
        
        problem_type = parsed_data.get("problem_type", "CVRP") # Used by helpers
        num_customer_nodes_original = len(original_instance_tuple_RAW[1])
        instance_specific_overall_dir = os.path.join(current_run_output_dir, instance_name_for_output); os.makedirs(instance_specific_overall_dir, exist_ok=True)
        
        # Normalize for NN-based methods (once per instance)
        instance_for_nn_processing = original_instance_tuple_RAW # Default to raw if no nodes
        depot_coords_raw = np.array(original_instance_tuple_RAW[0]); node_coords_raw = np.array(original_instance_tuple_RAW[1])
        min_coord_norm, range_coord_norm = None, None # For denormalization if needed by some scores
        if node_coords_raw.size > 0:
            all_coords_raw_for_norm = np.vstack((depot_coords_raw, node_coords_raw)); min_coord_norm = all_coords_raw_for_norm.min(axis=0); max_coord_norm = all_coords_raw_for_norm.max(axis=0)
            range_coord_norm = max_coord_norm - min_coord_norm; range_coord_norm[range_coord_norm == 0] = 1.0 # Avoid div by zero
            norm_depot = ((depot_coords_raw - min_coord_norm) / range_coord_norm).tolist(); norm_nodes = ((node_coords_raw - min_coord_norm) / range_coord_norm).tolist()
            instance_for_nn_processing = (norm_depot, norm_nodes, original_instance_tuple_RAW[2], original_instance_tuple_RAW[3])
            if len(original_instance_tuple_RAW) > 4: instance_for_nn_processing += original_instance_tuple_RAW[4:]
        time_parse_norm_sec = time.time() - time_parse_start

        # --- Generate Partitions (once per instance, if any partition method is used) ---
        raw_sequence_for_instance, initial_subproblems_for_instance = None, None
        time_partition_gen_sec = 0
        if any(m.startswith('Partition') for m in methods_to_run_final) and partitioner_model:
            logging.debug(f"  Generating partitions for {instance_name_for_output} using NN Partitioner...")
            part_gen_start_time = time.time()
            raw_sequence_for_instance, initial_subproblems_for_instance = generate_sequence_and_initial_routes_from_tuple(
                instance_for_nn_processing, problem_type, partitioner_model, device_nn
            )
            time_partition_gen_sec = time.time() - part_gen_start_time
            if raw_sequence_for_instance is None:
                logging.error(f"  Partition generation failed for {instance_basename}. Partition-based methods will be skipped for this instance.")
        
        # --- Iterate through each method for this instance ---
        for method_key in methods_to_run_final:
            logging.info(f"  -- Applying Method: {method_key} to {instance_name_for_output} --")
            current_method_results = {
                'problem_name': instance_name_for_output, 'problem_size_N': num_customer_nodes_original,
                'method_applied': method_key, 
                'cost': float('inf'),
                'time_parse_norm_sec': round(time_parse_norm_sec,4), 'time_partition_gen_sec': 0, 'time_merge_sec': 0,
                'time_solve_sec': 0, 'time_total_sec': 0, 'num_subproblems': 0
            }
            reconstructed_path_for_method = [] # Store the final path (original indices) for this method
            _internal_error_info = 'MethodNotRun' # Internal tracking, not for CSV

            # --- Direct Methods ---
            if method_key == 'DirectSolver':
                if not solver_model_nn: 
                    logging.warning(f"    Skipping {method_key}: NN Solver model not loaded.")
                    _internal_error_info = 'NNSolverNotLoaded'
                else:
                    print(f"    Applying Method: {method_key} to {instance_name_for_output}...") # CONSOLE
                    solve_res = run_direct_nn_solver_instance(instance_for_nn_processing, problem_type, solver_model_nn, device_nn, args.solver_aug_factor, sol_params)
                    _internal_normalized_score = solve_res.get('score', 'inf') 
                    current_method_results['time_solve_sec'] = round(solve_res['solve_time_seconds'],4)
                    reconstructed_path_for_method = solve_res['full_path_original_indices']
                    _internal_error_info = solve_res['error']

            elif method_key == 'DirectORTools':
                if not ortools_solve_vrp_func: 
                    logging.warning(f"    Skipping {method_key}: OR-Tools function not available.")
                    _internal_error_info = 'ORToolsNotLoaded'
                else:
                    print(f"    Applying Method: {method_key} to {instance_name_for_output}...") # CONSOLE
                    eff_tl_direct_or = get_effective_ortools_timelimit(num_customer_nodes_original, ORTOOLS_TIMELIMIT_MAP, 60) 
                    solve_res = run_direct_ortools_instance(original_instance_tuple_RAW, problem_type, eff_tl_direct_or, ortools_solve_vrp_func)
                    current_method_results['cost'] = solve_res['score'] 
                    current_method_results['time_solve_sec'] = round(solve_res['solve_time_seconds'],4)
                    reconstructed_path_for_method = solve_res['full_path_original_indices']
                    _internal_error_info = solve_res['error']
            
            # --- Partition-Based Methods ---
            elif method_key.startswith('Partition'):
                if raw_sequence_for_instance is None: 
                    logging.warning(f"    Skipping {method_key} for {instance_basename}: Partition generation failed.")
                    _internal_error_info = 'PartitionGenFailUpstream'
                else:
                    current_method_results['time_partition_gen_sec'] = round(time_partition_gen_sec,4)
                    merge_suffix = MERGE_CONFIG_TO_SUFFIX.get(method_key)
                    if not merge_suffix: 
                        logging.error(f"    Cannot determine merge suffix for {method_key}. Skipping.")
                        _internal_error_info = 'BadMergeSuffix'
                    else:
                        time_merge_start = time.time()
                        mc_num, mc_target = parse_mc_string(merge_suffix) 
                        merged_node_lists = merge_subproblems_by_centroid_fixed_size(
                            initial_subproblems=initial_subproblems_for_instance,
                            original_loc=instance_for_nn_processing[1],    
                            original_depot=instance_for_nn_processing[0][0], 
                            problem_size_for_dynamic_target=num_customer_nodes_original,
                            merge_num=mc_num,
                            target_node_count=mc_target,
                            problem_type=problem_type
                        )
                        current_method_results['time_merge_sec'] = round(time.time() - time_merge_start, 4)
                        current_method_results['num_subproblems'] = len(merged_node_lists)

                        if not merged_node_lists: 
                            logging.warning(f"    Merging for {method_key} on {instance_basename} yielded no subproblems. Score will be inf.")
                            _internal_error_info = 'NoMergedSubproblems'
                        else:
                            if method_key.startswith('PartitionSolver'): 
                                if not solver_model_nn: 
                                    logging.warning(f"    Skipping {method_key}: NN Solver model not loaded.")
                                    _internal_error_info = 'NNSolverNotLoaded'
                                else:
                                    print(f"    Applying Method: {method_key} to {instance_name_for_output}...") # CONSOLE
                                    solve_res = run_partition_nn_solver_instance(instance_for_nn_processing, problem_type, merged_node_lists, solver_model_nn, device_nn, args.solver_aug_factor, sol_params)
                                    _internal_normalized_score = solve_res.get('normalized_score', 'inf') 
                                    current_method_results['time_solve_sec'] = round(solve_res['time_solve_nn_sec'],4)
                                    reconstructed_path_for_method = solve_res['full_path_original_indices']
                                    _internal_error_info = solve_res['error']
                            
                            elif method_key.startswith('PartitionORTools'): 
                                if not ortools_solve_vrp_func: 
                                    logging.warning(f"    Skipping {method_key}: OR-Tools function not available.")
                                    _internal_error_info = 'ORToolsNotLoaded'
                                else:
                                    print(f"    Applying Method: {method_key} to {instance_name_for_output}...") # CONSOLE
                                    temp_subproblem_instance_tuples = []
                                    if merged_node_lists:
                                        depot_xy_raw = original_instance_tuple_RAW[0]
                                        nodes_xy_raw = original_instance_tuple_RAW[1]
                                        demands_raw = original_instance_tuple_RAW[2]
                                        capacity_raw = original_instance_tuple_RAW[3]
                                        num_orig_nodes_for_sub_tl = len(nodes_xy_raw)
                                        for node_indices_1based_list_tl in merged_node_lists:
                                            if not node_indices_1based_list_tl: continue
                                            valid_indices_for_sub_tl = [idx for idx in node_indices_1based_list_tl if 1 <= idx <= num_orig_nodes_for_sub_tl]
                                            if not valid_indices_for_sub_tl: continue
                                            sub_nodes_xy_tl = [nodes_xy_raw[i-1] for i in valid_indices_for_sub_tl]
                                            sub_demands_tl = [demands_raw[i-1] for i in valid_indices_for_sub_tl]
                                            temp_sub_instance_for_pad = (depot_xy_raw, sub_nodes_xy_tl, sub_demands_tl, capacity_raw)
                                            temp_subproblem_instance_tuples.append(temp_sub_instance_for_pad)
                                    eff_tl_sub_or = 10 
                                    if temp_subproblem_instance_tuples:
                                        _padded_subs_for_tl, actual_pad_size_for_subs_tl = pad_subproblem_batch(
                                            temp_subproblem_instance_tuples, problem_type 
                                        )
                                        if actual_pad_size_for_subs_tl > 0:
                                            eff_tl_sub_or = get_effective_ortools_timelimit(actual_pad_size_for_subs_tl, ORTOOLS_TIMELIMIT_MAP, 10) 
                                            logging.info(f"    PartitionORTools: Subproblem padded size N_sub_pad={actual_pad_size_for_subs_tl}, using OR-Tools timelimit: {eff_tl_sub_or}s for subproblems.")
                                        else:
                                            logging.warning("    PartitionORTools: Could not determine subproblem padded size for timelimit. Using default 10s.")
                                    else:
                                         logging.warning("    PartitionORTools: No valid subproblem instances to determine dynamic timelimit. Using default 10s.")
                                    method_run_result = run_partition_ortools_instance(
                                        original_instance_tuple_RAW, problem_type, merged_node_lists, eff_tl_sub_or, 
                                        ortools_solve_vrp_func, args.ortools_subproblem_workers,
                                        args.ortools_stagnation_duration, args.ortools_min_improvement_pct 
                                    )
                                    current_method_results['cost'] = method_run_result.get('score', 'inf')
                                    current_method_results['time_solve_sec'] = round(method_run_result.get('solve_time_seconds', 0), 4)
                                    current_method_results['num_subproblems'] = method_run_result.get('num_subproblems', 0)
                                    reconstructed_path_for_method = method_run_result.get('full_path_original_indices', []) 
                                    _internal_error_info = method_run_result.get('error', None) 
            else:
                logging.warning(f"    Method {method_key} logic not implemented. Skipping.")
                _internal_error_info = 'MethodNotImplemented'
            
            # --- Calculate total time for all methods AFTER specific timings are set ---
            solve_time_for_total = 0
            if method_key.startswith('PartitionORTools'):
                # For PartitionORTools, adjust solve time for parallelism
                time_solve_component = current_method_results.get('time_solve_sec', 0)
                num_subs = current_method_results.get('num_subproblems', 0)
                workers_for_calc = args.ortools_subproblem_workers if args.ortools_subproblem_workers > 0 else (os.cpu_count() or 1)
                workers_for_calc = min(workers_for_calc, num_subs if num_subs > 0 else 1, os.cpu_count() or 1)
                batches = 1
                if num_subs > 0 and workers_for_calc > 0: batches = math.ceil(num_subs / workers_for_calc)
                if batches < 1: batches = 1 # Safety
                solve_time_for_total = time_solve_component / batches
            else:
                solve_time_for_total = current_method_results.get('time_solve_sec', 0)

            current_method_results['time_total_sec'] = round(
                current_method_results.get('time_parse_norm_sec', 0) + 
                current_method_results.get('time_partition_gen_sec', 0) + 
                current_method_results.get('time_merge_sec', 0) + 
                solve_time_for_total, 
            4)
            
            # --- Final calculations for this method (cost for NN, ensure defaults) ---
            if (method_key.startswith('DirectSolver') or method_key.startswith('PartitionSolver')) and reconstructed_path_for_method:
                current_method_results['cost'] = calculate_path_cost_original_coords(reconstructed_path_for_method, original_instance_tuple_RAW[0], original_instance_tuple_RAW[1])
            elif not reconstructed_path_for_method and current_method_results['cost'] == float('inf') and not method_key.startswith('PartitionORTools') and not method_key.startswith('DirectORTools') :
                # Ensure it remains inf if path reconstruction failed for NN methods
                # For OR-Tools methods, cost is already set from their 'score'
                current_method_results['cost'] = float('inf')
            
            # Ensure all time fields are present and are floats for consistent CSV output, even if method skipped early
            for time_key in ['time_parse_norm_sec', 'time_partition_gen_sec', 'time_merge_sec', 'time_solve_sec', 'time_total_sec']:
                if time_key not in current_method_results or not isinstance(current_method_results[time_key], (int, float)):
                    current_method_results[time_key] = 0.0
            if 'num_subproblems' not in current_method_results or not isinstance(current_method_results['num_subproblems'], int):
                current_method_results['num_subproblems'] = 0

            all_results_for_csv.append(current_method_results)
            
            # Save detailed output for this method
            method_output_dir = os.path.join(instance_specific_overall_dir, method_key); os.makedirs(method_output_dir, exist_ok=True)
            detail_json_path = os.path.join(method_output_dir, f"{instance_name_for_output}_{method_key}_details.json")
            with open(detail_json_path, 'w') as f_json: json.dump(current_method_results, f_json, indent=4)
            if reconstructed_path_for_method:
                path_pkl_path = os.path.join(method_output_dir, f"{instance_name_for_output}_{method_key}_path.pkl")
                with open(path_pkl_path, 'wb') as f_pkl: pickle.dump(reconstructed_path_for_method, f_pkl)

            # Logging the result for this method
            log_cost_orig_str = f"{current_method_results['cost']:.2f}" if isinstance(current_method_results['cost'], (int, float)) else str(current_method_results['cost'])
            log_total_time_str = f"{current_method_results['time_total_sec']:.2f}" if isinstance(current_method_results['time_total_sec'], (int, float)) else str(current_method_results['time_total_sec'])

            logging.info(f"    Method {method_key} -> Cost(Orig): {log_cost_orig_str}, TotalTime: {log_total_time_str}s, Error: {_internal_error_info}")
            if not args.verbose_console: # Print summary to console if not verbose (which would already show the above INFO)
                 print(f"      -> Cost(Orig): {log_cost_orig_str}, TotalTime: {log_total_time_str}s, Error: {_internal_error_info}")

        # --- Attempt to add Optimal Solution from .sol file (MOVED HERE - outside method loop, inside instance loop) ---
        sol_file_path = vrp_file_path.replace(".vrp", ".sol")
        if os.path.exists(sol_file_path):
            try:
                with open(sol_file_path, 'r') as f_sol:
                    optimal_cost = None
                    for line in f_sol:
                        line_lower = line.lower()
                        if 'cost' in line_lower: # More robust check for "Cost" or "cost"
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part.lower() == 'cost' and i + 1 < len(parts):
                                    try:
                                        optimal_cost = float(parts[i+1])
                                        break # Found cost, exit inner loop
                                    except ValueError:
                                        logging.warning(f"Found 'cost' in {sol_file_path} but could not parse value: {parts[i+1]}")
                                        optimal_cost = float('inf') # Indicate parsing failure
                                        break
                            if optimal_cost is not None: # If cost found (or parse error for cost)
                                break # Exit outer loop (line iteration)
                    
                    if optimal_cost is not None and optimal_cost != float('inf'):
                        optimal_sol_entry = {
                            'problem_name': instance_name_for_output,
                            'problem_size_N': num_customer_nodes_original,
                            'method_applied': "OptimalSolution",
                            'cost': optimal_cost, # Ensure this matches CSV header
                            'time_parse_norm_sec': 0.0,
                            'time_partition_gen_sec': 0.0,
                            'time_merge_sec': 0.0,
                            'time_solve_sec': 0.0,
                            'time_total_sec': 0.0,
                            'num_subproblems': 0
                        }
                        all_results_for_csv.append(optimal_sol_entry)
                        logging.info(f"Added OptimalSolution entry for {instance_name_for_output} from {sol_file_path}, Cost: {optimal_cost}")
                    elif optimal_cost == float('inf'):
                         logging.warning(f"Optimal solution cost could not be parsed from {sol_file_path}. OptimalSolution entry will not be added.")
                    else:
                        logging.warning(f"Could not find 'Cost' line in {sol_file_path}. OptimalSolution entry will not be added.")

            except Exception as e_sol:
                logging.warning(f"Error reading or parsing .sol file {sol_file_path}: {e_sol}. OptimalSolution entry will not be added.")
        else:
            logging.info(f".sol file not found for {instance_name_for_output} at {sol_file_path}. OptimalSolution entry will not be added.")
        # --- End of Optimal Solution addition ---

        # --- End Instance Loop (VRP files) ---
    logging.info(f"Finished processing all {len(vrp_files)} VRP files.")

    # --- Write Final CSV ---
    try:
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers); writer.writeheader(); writer.writerows(all_results_for_csv)
        logging.info(f"Batch multi-method processing complete. Summary CSV saved to: {csv_file_path}")
    except IOError: logging.error(f"Error writing CSV to {csv_file_path}", exc_info=True)

if __name__ == "__main__":
    # Needed for OR-Tools on some systems if using ProcessPoolExecutor later
    if os.name == 'nt': # Setting start method is often needed on Windows
        # Check if running in a context where it's already set (e.g. certain IDEs/notebooks)
        if mp.get_start_method(allow_none=True) is None:
            try: 
                mp.set_start_method('spawn', force=True)
                # Use a print here as logging might not be fully set up if this fails early
                print("INFO: Multiprocessing start method set to 'spawn'.")
            except RuntimeError as e:
                print(f"WARNING: Could not set multiprocessing start method to 'spawn': {e}. This might lead to issues with OR-Tools parallelism.")
    main_batch_processor() 