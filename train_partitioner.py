import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import logging
import pytz
import math
import random
import re
import gc
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict, Counter
import numpy as np
from functools import partial

from utils import (load_dataset, get_env, AverageMeter, TimeEstimator,
                   seed_everything, VRP_DATA_FORMAT, DATASET_PATHS, get_model, num_param)
from models import MOEModel, MOEModel_Light
from partitioner_solver_utils import (
    load_moe_model, _split_sequence_by_zeros, merge_subproblems_by_centroid_fixed_size,
    create_subproblem_instance, pad_subproblem_batch,
    prepare_batch_tensor_data, solve_vrp_batch,
    DEFAULT_MODEL_PARAMS
)

# Initialize logger at module level
logger = logging.getLogger(__name__)

# Define the problems included in Train_ALL (consistent with train.py)
TRAIN_ALL_PROBLEMS = ["CVRP", "OVRP", "VRPB", "VRPL", "VRPTW"]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Implicit Partitioner (Solver Arch) using REINFORCE")

    # --- Data Parameters ---
    parser.add_argument('--problem', type=str, default="CVRP", 
                        help=f"Original VRP problem type or 'Train_ALL' for multi-task training. Train_ALL includes: {TRAIN_ALL_PROBLEMS}")
    parser.add_argument('--problem_size', type=int, default=100, help="Size of the original VRP instances (currently assumes same size for Train_ALL)")
    parser.add_argument('--validation_size', type=int, default=100, help="Number of instances *per problem type* from the dataset to use for validation")
    parser.add_argument('--pomo_size', type=int, default=1, help="POMO size for the model being trained (usually 1 for this reward scheme)")

    # --- Model Parameters ---
    parser.add_argument('--model_type', type=str, default="MOE", choices=["SINGLE", "MTL", "MOE", "MOE_LIGHT"], help="Type of model architecture to train")
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--sqrt_embedding_dim', type=float, default=128**(1/2))
    parser.add_argument('--encoder_layer_num', type=int, default=6, help="the number of MHA in encoder")
    parser.add_argument('--decoder_layer_num', type=int, default=1, help="the number of MHA in decoder")
    parser.add_argument('--qkv_dim', type=int, default=16)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--logit_clipping', type=float, default=10)
    parser.add_argument('--ff_hidden_dim', type=int, default=512)
    parser.add_argument('--num_experts', type=int, default=4, help="the number of FFN in a MOE layer (if model_type is MOE/MOE_LIGHT)")
    parser.add_argument('--eval_type', type=str, default="softmax", choices=["argmax", "softmax"], help="Evaluation type during training (softmax for sampling in REINFORCE)")
    parser.add_argument('--norm', type=str, default="instance", choices=["batch", "batch_no_track", "instance", "layer", "rezero", "none"])
    parser.add_argument('--norm_loc', type=str, default="norm_last", choices=["norm_first", "norm_last"], help="whether conduct normalization before MHA/FFN/MOE")
    parser.add_argument('--topk', type=int, default=2, help="how many experts (on average) to route for each input (if MoE)")
    parser.add_argument('--expert_loc', type=str, nargs='+', default=['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'], help="where to use MOE (if MoE)")
    parser.add_argument('--routing_level', type=str, default="node", choices=["node", "instance", "problem"], help="routing level for MOE (if MoE)")
    parser.add_argument('--routing_method', type=str, default="input_choice", choices=["input_choice", "expert_choice", "soft_moe", "random"], help="routing method for MOE (if MoE)")
    # parser.add_argument('--part_use_tw_features', action='store_true', help='Include TW features in input') # Implicitly handled by env/data format

    # --- Model Loading ---
    parser.add_argument('--model_checkpoint_start', type=str, default=None, help="Path to pre-trained Solver/MOE model to start/fine-tune from (optional)")

    # --- Solver Model Parameters (Fixed, for reward calculation AND baseline) --- # Updated help
    parser.add_argument('--solver_checkpoint', type=str, default="pretrained/mvmoe_8e_n50/epoch-2500.pt", help="Path to the FIXED pre-trained solver model checkpoint (REQUIRED for reward AND baseline)") # Updated help
    # Rename solver_native_size to adaptive_merge_target_size
    parser.add_argument('--adaptive_merge_target_size', type=int, default=0, help="Target node count for adaptive subproblem merging (if merge_num <= 0). <=0 for dynamic selection based on problem size.")
    parser.add_argument('--solver_aug_factor', type=int, default=8, help="Augmentation factor for FIXED solver during reward calculation AND baseline") # Updated help

    # --- Training Parameters ---
    parser.add_argument('--epochs', type=int, default=100, help="Total training epochs")
    parser.add_argument('--epoch_size', type=int, default=10000, help="Number of instances to process per epoch")
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--train_batch_size', type=int, default=64, help="Batch size for processing instances (influences grad accum)") # Added batch size
    parser.add_argument('--grad_accum_steps', type=int, default=1, help="Gradient accumulation steps") # Keep grad accum
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, help="Max norm for gradient clipping (0 to disable)")
    parser.add_argument('--ema_beta', type=float, default=0.9, help="DEPRECATED: Decay factor for EMA baseline reward (now uses greedy rollout baseline)")
    parser.add_argument('--moe_loss_coef', type=float, default=1e-2, help="Coefficient for MoE load balancing loss (if applicable)")
    parser.add_argument('--milestones', type=int, nargs='+', default=[80, 95], help='Epochs at which to decay learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate decay factor')
    parser.add_argument('--checkpoint_dir', type=str, default='./results/TAM_MoE', help="Base directory to save implicit partitioner checkpoints")
    parser.add_argument('--log_dir', type=str, default='./results/TAM_MoE', help="Base directory to save logs")
    parser.add_argument('--validation_interval', type=int, default=5)
    parser.add_argument('--model_save_interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--reward_fail_penalty', type=float, default=1e7, help='Penalty if reward calculation fails')
    parser.add_argument('--resume', type=str, default=None, help="Path to implicit partitioner checkpoint to RESUME training from")
    parser.add_argument('--verbose_log', action='store_true', help="Print detailed logs to console if set")

    # --- Subproblem Parameters ---
    parser.add_argument('--merge_num', type=int, default=-1, help="Number of consecutive raw subproblems to merge for reward calculation")

    # --- Validation Parameters ---
    parser.add_argument('--validate_batch_size', type=int, default=None, help="Batch size for validation. Defaults to train_batch_size if not set.")
    parser.add_argument('--generalization_validation_interval', type=int, default=None, help="Frequency (in epochs) to run generalization validation on larger sizes (0 to disable). Defaults to the value of --validation_interval if not set.")
    parser.add_argument('--generalization_val_samples', type=int, default=10, help="Number of samples per size/problem for generalization validation")
    parser.add_argument('--generalization_baseline_aug_factor', type=int, default=1, help="Augmentation factor for FIXED solver during large-scale baseline calculation (to save memory)")
    parser.add_argument('--generalization_sizes', type=int, nargs='+', default=[500, 1000, 2000], help="Sizes to run generalization validation on (if --generalization_validation_interval is set)")
    # Beam search width for validation/inference
    parser.add_argument('--beam_size', type=int, default=1, help="Beam size for validation (uses POMO-style execution). If 1, equivalent to greedy search.")

    # Data Augmentation
    parser.add_argument('--apply_data_augmentation', action='store_true', help="Apply simple geometric and demand augmentation during training.")
    parser.add_argument('--demand_perturb_factor', type=float, default=0.1, help="Max percentage for demand perturbation (e.g., 0.1 for +/-10%).")

    args = parser.parse_args()
    return args

def calculate_partition_reward(raw_sequence, original_instance_tuple, original_loc,
                               problem_type, merge_num, solver_model, solver_env_class,
                               problem_size_for_dynamic_target: int, # Added
                               adaptive_merge_target_from_args: int, # Added
                               solver_aug_factor, device, fail_penalty):
    """
    Calculates the reward for a partition sequence by solving the generated subproblems.
    Uses centroid-based merging.

    Args:
        raw_sequence (list[int]): Sequence from the TRAINED model.
        original_instance_tuple (tuple): Original VRP data.
        original_loc (list[list[float]]): Coordinates of original customer nodes.
        problem_type (str): Original VRP type.
        merge_num (int): How many raw subproblems to merge based on centroid proximity.
        solver_model (nn.Module): Pre-loaded FIXED solver model.
        solver_env_class (type): Environment class for the FIXED solver.
        problem_size_for_dynamic_target (int): N of the original problem for dynamic merge target selection.
        adaptive_merge_target_from_args (int): Value from --adaptive_merge_target_size CLI arg.
        solver_aug_factor (int): Augmentation factor for solving with FIXED solver.
        device (torch.device): Computation device.
        fail_penalty (float): Penalty value if any step fails.

    Returns:
        tuple(float, list[list[int]]): Negative total cost and list of final merged subproblem node lists.
                                       Returns (-penalty, []) on major failure.
    """
    try:
        # Helper to remove padding zeros remains the same
        def remove_padding_zeros(path_list):
            start = 0
            while start < len(path_list) and path_list[start] == 0:
                start += 1
            end = len(path_list) - 1
            while end >= 0 and path_list[end] == 0:
                end -= 1
            return path_list[start:end+1] if start <= end else []

        # --- Extract original depot location ---
        try:
            original_depot = np.array(original_instance_tuple[0])
            if original_depot.ndim > 1:
                original_depot = original_depot.flatten()
        except (IndexError, TypeError, ValueError) as e:
            logger.error(f"Reward Calc: Could not extract depot location: {e}. Returning penalty.")
            return -fail_penalty, []

        # 1. Split sequence into initial subproblems
        initial_subproblems = _split_sequence_by_zeros(raw_sequence)
        if not initial_subproblems:
            logger.debug("Reward Calc: No initial subproblems extracted from raw sequence: %s", raw_sequence)
            return -fail_penalty, [] 

        # 2. Merge subproblems
        merged_node_lists = merge_subproblems_by_centroid_fixed_size(
            initial_subproblems,
            original_loc,      
            original_depot,    
            problem_size_for_dynamic_target=problem_size_for_dynamic_target, # Added
            merge_num=merge_num,
            target_node_count=adaptive_merge_target_from_args # Added
        )
        if not merged_node_lists:
             logger.warning(f"Reward Calc: Weighted polar merging resulted in no final subproblems from {len(initial_subproblems)} initial ones.")
             return -fail_penalty, [] 

        # 3. Create Subproblem Instance Tuples from merged lists
        subproblem_instance_tuples = []
        num_original_nodes = len(original_loc)
        for node_indices in merged_node_lists:
            # Validate indices before creating instance
            valid_indices = [idx for idx in node_indices if 1 <= idx <= num_original_nodes]
            if not valid_indices:
                 logger.warning(f"Reward Calc: Merged list {node_indices} contains no valid customer nodes.")
                 continue # Skip this empty/invalid merged list
            sub_instance = create_subproblem_instance(original_instance_tuple, problem_type, valid_indices)
            if sub_instance:
                subproblem_instance_tuples.append(sub_instance)
            else:
                logger.warning(f"Reward Calc: Failed to create subproblem instance for indices: {valid_indices}")
                # Decide if failure to create one subproblem invalidates the whole reward
                # For robustness, let's apply penalty and return immediately
                return -fail_penalty, merged_node_lists # Return penalty and the problematic merged list

        if not subproblem_instance_tuples:
            logger.warning("Reward Calc: No subproblem instances could be created from merged lists.")
            return -fail_penalty, merged_node_lists # Return penalty and the merged lists

        # 4. Pad Subproblem Batch (unchanged)
        padded_batch_tuples, target_pad_size = pad_subproblem_batch(subproblem_instance_tuples, problem_type)
        if not padded_batch_tuples:
            logger.error("Reward Calc: Padding subproblem batch failed.")
            return -fail_penalty, merged_node_lists

        # 5. Prepare Solver Input Tensors (unchanged)
        padded_batch_tensor_data = prepare_batch_tensor_data(
            padded_batch_tuples, problem_type, device
        )
        if not padded_batch_tensor_data:
            logger.error("Reward Calc: Preparing batch tensor data failed.")
            return -fail_penalty, merged_node_lists

        # 6. Solve Subproblem Batch using the FIXED solver (unchanged)
        solver_results = solve_vrp_batch(
            solver_model=solver_model, # Pass the FIXED solver here
            solver_env_class=solver_env_class,
            original_instance_tuples=subproblem_instance_tuples, # Pass created subproblem tuples
            padded_batch_data=padded_batch_tensor_data,
            padded_problem_size=target_pad_size,
            problem_type=problem_type,
            device=device,
            aug_factor=solver_aug_factor
        )

        # 7. Aggregate Costs (unchanged, but logs use merged_node_lists on failure)
        total_cost = 0.0
        if not solver_results or len(solver_results) != len(subproblem_instance_tuples):
            logger.error(f"Reward Calc: Solver results mismatch. Expected {len(subproblem_instance_tuples)}, got {len(solver_results) if solver_results else 0}.")
            return -fail_penalty * max(1, len(subproblem_instance_tuples)), merged_node_lists

        num_failed_subproblems = 0
        for aug_score, _ in solver_results:
            if aug_score == float('inf') or aug_score is None or math.isnan(aug_score):
                total_cost += fail_penalty
                num_failed_subproblems += 1
            else:
                total_cost += aug_score

        if num_failed_subproblems > 0:
             logger.debug(f"Reward Calc: {num_failed_subproblems}/{len(solver_results)} subproblems failed solver, applied penalty.")

        reward = -total_cost
        # Return reward and the final merged node lists
        return reward, merged_node_lists 

    except Exception as e:
        logger.exception(f"Error during reward calculation: {e}")
        return -fail_penalty, []

# --- Helper function to solve a full instance ---
def _solve_full_instance(instance_tuple, problem_type, solver_model, solver_env_class, args, device, validation_problem_size, aug_factor):
    """Solves a single full VRP instance using the fixed solver.
    
    Args:
        instance_tuple: The VRP instance data.
        problem_type: The VRP problem type.
        solver_model: The pre-loaded fixed solver model.
        solver_env_class: The environment class for the solver.
        args: Command line arguments.
        device: Computation device.
        validation_problem_size: The actual size N of the instance being solved.
        aug_factor: The augmentation factor to use for this solve.
    Returns:
        float: The reward (negative cost) or negative penalty on failure.
    """
    try:
        # Use the provided solver_env_class for this specific problem type
        if not solver_env_class:
             raise ValueError(f"Missing solver_env_class for problem_type: {problem_type}")
             
        # Pad the instance to solver's native size
        padded_batch_tuples, target_pad_size = pad_subproblem_batch([instance_tuple], problem_type)
        if not padded_batch_tuples:
            # Log the specific size that failed padding
            logger.error(f"Failed to pad full instance (type: {problem_type}, size: {validation_problem_size})")
            return -args.reward_fail_penalty # Return negative penalty as reward

        # Prepare tensor data
        padded_batch_tensor_data = prepare_batch_tensor_data(
            padded_batch_tuples, problem_type, device
        )
        if not padded_batch_tensor_data:
             logger.error(f"Failed to prepare tensor data for full instance (type: {problem_type}, size: {validation_problem_size})")
             return -args.reward_fail_penalty # Return negative penalty as reward

        # Solve using the fixed solver with the specified aug_factor
        solver_results = solve_vrp_batch(
            solver_model=solver_model,
            solver_env_class=solver_env_class,
            original_instance_tuples=[instance_tuple], # Pass original tuple in a list
            padded_batch_data=padded_batch_tensor_data,
            padded_problem_size=target_pad_size, # This is the size AFTER padding
            problem_type=problem_type,
            device=device,
            aug_factor=aug_factor # Use the passed aug_factor
        )

        # Extract cost
        if solver_results and solver_results[0][0] != float('inf') and solver_results[0][0] is not None and not math.isnan(solver_results[0][0]):
            # Reward is negative cost
            return -solver_results[0][0] 
        else:
            logger.warning(f"Solver failed to solve full instance (type: {problem_type}, size: {validation_problem_size}). Returning high penalty reward.")
            return -args.reward_fail_penalty # Return negative penalty as reward

    except Exception as e:
        logger.exception(f"Error solving full instance (type: {problem_type}, size: {validation_problem_size}): {e}")
        return -args.reward_fail_penalty

# --- Validation Function --- 
def _run_validation_for_problem(model_being_trained, fixed_solver_model, all_env_classes, current_validation_data, args, device, current_epoch, current_problem_type, validation_problem_size):
    """
    Runs validation for a specific problem type and size.
    Args:
        model_being_trained: The model currently being trained.
        fixed_solver_model: The fixed solver model for reward/baseline.
        all_env_classes (dict): Dictionary mapping problem type string to EnvClass.
        current_validation_data (list): List of instance tuples for the current problem type and size.
        args: Command line arguments.
        device: Computation device.
        current_epoch (int): The current training epoch number for logging.
        current_problem_type (str): The specific problem type being validated.
        validation_problem_size (int): The specific problem size (N) being validated.
    Returns:
        dict or None: Dictionary containing validation statistics, or None on failure.
    """
    logger.info(f"Starting validation for {current_problem_type} N={validation_problem_size}...")
    if not current_validation_data:
        logging.warning(f"No validation data provided for {current_problem_type} N={validation_problem_size}, skipping.")
        return None

    model_being_trained.eval() # Set model to eval mode
    fixed_solver_model.eval() # Fixed solver is always eval

    partitioned_reward_meter = AverageMeter()
    baseline_reward_meter = AverageMeter() # For direct solve reward
    seq_len_meter = AverageMeter()
    # For subproblem stats from the best beam
    best_beam_subproblem_counts = []
    best_beam_subproblem_node_counts = []

    # Get the correct EnvClass for solver
    try:
        CurrentSolverEnvClass = all_env_classes[current_problem_type]
    except KeyError:
        logger.error(f"No EnvClass found for {current_problem_type}. Skipping validation for N={validation_problem_size}.")
        return None

    num_val_instances = len(current_validation_data)
    # Start with the global validate_batch_size from args
    effective_validate_batch_size = args.validate_batch_size

    # For generalization validation of very large problem sizes, dynamically cap the batch size
    # The 'validation_problem_size' argument to this function tells us the N for the current validation run
    if validation_problem_size >= 1000: # Threshold for large problems
        if validation_problem_size == 1000:
            cap_batch_size_for_large_N = effective_validate_batch_size // 2
        elif validation_problem_size == 2000:
            cap_batch_size_for_large_N = effective_validate_batch_size // 4
        elif validation_problem_size == 5000:
            cap_batch_size_for_large_N = effective_validate_batch_size // 8
        logger.info(f"Validation for N={validation_problem_size}: Dynamically capping effective batch size from {effective_validate_batch_size} to {cap_batch_size_for_large_N}.")
        effective_validate_batch_size = cap_batch_size_for_large_N

    # Ensure effective_validate_batch_size is at least 1 if there are instances
    if num_val_instances > 0 and effective_validate_batch_size <= 0:
        logger.warning(f"Validation for N={validation_problem_size}: Effective batch size ({effective_validate_batch_size}) is <= 0. Setting to 1.")
        effective_validate_batch_size = 1
    elif num_val_instances == 0: # No data, no batches
        effective_validate_batch_size = 1 # Placeholder, loop won't run

    num_val_batches = math.ceil(num_val_instances / effective_validate_batch_size) if effective_validate_batch_size > 0 else 0

    # --- Calculate Baseline Rewards (Direct Solve) ---
    baseline_aug_factor = args.generalization_baseline_aug_factor if validation_problem_size >= 500 else args.solver_aug_factor
    logger.info(f"Calculating baseline rewards for {current_problem_type} N={validation_problem_size} (Aug Factor: {baseline_aug_factor}, Batch Size: {effective_validate_batch_size})...")

    if num_val_batches > 0:
        pbar_baseline = tqdm(range(num_val_batches), desc=f"Baseline {current_problem_type}_N{validation_problem_size}", unit="batch", leave=False)
        for batch_idx in pbar_baseline:
            start_idx = batch_idx * effective_validate_batch_size
            end_idx = min((batch_idx + 1) * effective_validate_batch_size, num_val_instances)
            current_batch_tuples_baseline = current_validation_data[start_idx:end_idx]
            current_actual_batch_s = len(current_batch_tuples_baseline)

            if current_actual_batch_s == 0:
                continue

            try:
                padded_batch_b, target_pad_size_b = pad_subproblem_batch(current_batch_tuples_baseline, current_problem_type)
                if not padded_batch_b:
                    raise ValueError("Failed to pad baseline batch")

                tensor_data_b = prepare_batch_tensor_data(padded_batch_b, current_problem_type, device)
                if not tensor_data_b:
                    raise ValueError("Failed to prepare tensor data for baseline batch")

                solver_results_b = solve_vrp_batch(
                    solver_model=fixed_solver_model,
                    solver_env_class=CurrentSolverEnvClass,
                    original_instance_tuples=current_batch_tuples_baseline,
                    padded_batch_data=tensor_data_b,
                    padded_problem_size=target_pad_size_b,
                    problem_type=current_problem_type,
                    device=device,
                    aug_factor=baseline_aug_factor
                )

                if solver_results_b and len(solver_results_b) == current_actual_batch_s:
                    for res_idx, (cost, _) in enumerate(solver_results_b):
                        if cost != float('inf') and cost is not None and not math.isnan(cost):
                            baseline_reward_meter.update(-cost)
                        else:
                            logger.warning(f"Solver failed for instance {start_idx + res_idx} in baseline batch N={validation_problem_size}. Applying penalty.")
                            baseline_reward_meter.update(-args.reward_fail_penalty)
                else:
                    logger.error(f"Baseline results mismatch for batch {batch_idx} (N={validation_problem_size}). Expected {current_actual_batch_s}, got {len(solver_results_b) if solver_results_b else 0}. Applying penalties.")
                    for _ in range(current_actual_batch_s):
                        baseline_reward_meter.update(-args.reward_fail_penalty)
            except Exception as e:
                logger.error(f"Error during baseline batch {batch_idx} for N={validation_problem_size}: {e}. Applying penalties.")
                for _ in range(current_actual_batch_s):
                    baseline_reward_meter.update(-args.reward_fail_penalty)
            
            pbar_baseline.set_postfix(avg_baseline_reward=f"{baseline_reward_meter.avg:.2f}")
    else: # No batches to run, e.g. no validation data
        logger.info(f"No baseline batches to run for {current_problem_type} N={validation_problem_size}.")


    avg_baseline_reward = baseline_reward_meter.avg if baseline_reward_meter.count > 0 else -args.reward_fail_penalty


    # --- Calculate Partitioned Rewards and Stats --- 
    logger.info(f"Calculating partitioned rewards for {current_problem_type} N={validation_problem_size} (Model trained on N={args.problem_size}, Beam Size: {args.beam_size}, Val Batch Size: {effective_validate_batch_size})...")
    
    # Prepare environment class for the model being trained (using validation_problem_size)
    try:
        TrainEnvClass = get_env(current_problem_type)[0] # This is the env for the model being trained
    except Exception as e:
        logger.error(f"Failed to get Env class {current_problem_type} for validation N={validation_problem_size}: {e}")
        # If env class fails, cannot proceed with partitioned rewards.
        # Populate stats with defaults indicating failure if baseline was calculated.
        if baseline_reward_meter.count > 0: # Baseline was calculated
             partitioned_cost = -(-args.reward_fail_penalty * num_val_instances if num_val_instances > 0 else 0) # Penalized cost
             baseline_cost = -avg_baseline_reward
             gap_percent = ((baseline_cost - partitioned_cost) / baseline_cost) * 100 if baseline_cost > 1e-9 else float('nan')
             logger.warning("Partitioned reward calculation skipped due to Env class error. Reporting with max penalty.")
             return {
                'validation_problem_size': validation_problem_size,
                'partitioned_reward': -args.reward_fail_penalty * num_val_instances if num_val_instances > 0 else 0,
                'baseline_reward': avg_baseline_reward,
                'gap_percent': gap_percent,
                'avg_seq_len': 0, 'avg_subproblems': 0, 'subproblem_count_dist': Counter(),
                'avg_nodes_per_subproblem': 0, 'subproblem_node_dist': Counter()
            }
        return None # Critical failure

    env_params = {"problem_size": validation_problem_size, "pomo_size": args.beam_size}

    if num_val_batches > 0:
        pbar_val = tqdm(range(num_val_batches), desc=f"Part.{current_problem_type}_N{validation_problem_size}", unit="batch")
        with torch.no_grad():
            for batch_idx in pbar_val:
                start_idx = batch_idx * effective_validate_batch_size
                end_idx = min((batch_idx + 1) * effective_validate_batch_size, num_val_instances)
                current_batch_tuples_val = current_validation_data[start_idx:end_idx]
                current_actual_val_batch_s = len(current_batch_tuples_val)

                if current_actual_val_batch_s == 0:
                    continue

                all_batch_beam_sequences_tensor = None
                try:
                    val_env = TrainEnvClass(**env_params)
                    padded_val_batch, target_pad_size_val = pad_subproblem_batch(current_batch_tuples_val, current_problem_type)
                    if not padded_val_batch:
                        raise ValueError("Failed to pad validation batch")
                    
                    val_tensor_data = prepare_batch_tensor_data(padded_val_batch, current_problem_type, device)
                    if not val_tensor_data:
                        raise ValueError("Failed to prepare tensor data for validation batch")
                    
                    val_env.load_problems(batch_size=current_actual_val_batch_s, problems=val_tensor_data, aug_factor=1)
                    
                    reset_state, _, _ = val_env.reset()
                    model_being_trained.pre_forward(reset_state)
                    model_being_trained.set_eval_type('argmax')

                    state, _, done = val_env.pre_step()
                    while not done:
                        selected, _ = model_being_trained(state)
                        state, _, done = val_env.step(selected)

                    if hasattr(val_env, 'selected_node_list') and val_env.selected_node_list is not None:
                        if val_env.selected_node_list.numel() > 0:
                            # Expected shape: (current_actual_val_batch_s, args.beam_size, seq_len)
                            all_batch_beam_sequences_tensor = val_env.selected_node_list.view(current_actual_val_batch_s, args.beam_size, -1)
                        else:
                            logger.warning(f"Validation N={validation_problem_size}, Batch {batch_idx}: env.selected_node_list is empty.")
                    else:
                        logger.warning(f"Validation N={validation_problem_size}, Batch {batch_idx}: env.selected_node_list not found or None.")

                    if all_batch_beam_sequences_tensor is None or all_batch_beam_sequences_tensor.shape[0] != current_actual_val_batch_s:
                        logger.error(f"Sequence generation failed or shape mismatch for val batch {batch_idx} (N={validation_problem_size}). Penalizing instances in batch.")
                        for _ in range(current_actual_val_batch_s):
                            partitioned_reward_meter.update(-args.reward_fail_penalty)
                            seq_len_meter.update(0)
                            best_beam_subproblem_counts.append(0)
                        continue # to next batch processing

                    # --- Parallelized Beam Evaluation --- 
                    all_subproblems_global = [] # Stores (subproblem_instance_tuple, problem_type_for_subproblem)
                    # Tracks ((instance_idx_in_batch, beam_idx), num_subproblems, raw_sequence_len, original_instance_tuple, original_loc_val, merged_node_lists_for_beam)
                    beam_metadata_map = {}

                    for i in range(current_actual_val_batch_s): # Iterate through instances in the current validation batch
                        instance_tuple = current_batch_tuples_val[i]
                        original_loc_val = None
                        original_depot = None
                        
                        try:
                            # Attempt to extract critical data for subproblem generation
                            original_loc_val = instance_tuple[1]
                            original_depot = np.array(instance_tuple[0]).flatten()
                        except (IndexError, TypeError, ValueError) as e:
                            logger.error(f"Data extraction failed for instance {start_idx + i} (N={validation_problem_size}): {e}. All beams for this instance will be penalized.")
                            # Mark all beams for this instance as penalized and skip to the next instance
                            for beam_idx in range(args.beam_size):
                                beam_id = (i, beam_idx)
                                beam_metadata_map[beam_id] = (0, 0, instance_tuple, None, [], True) # num_subproblems, raw_seq_len, ..., is_penalized
                            continue # To the next instance i in the batch
                        
                        # If we reach here, original_loc_val and original_depot are valid for instance i
                        for beam_idx in range(args.beam_size):
                            beam_id = (i, beam_idx)
                            beam_raw_sequence_list = all_batch_beam_sequences_tensor[i, beam_idx, :].cpu().tolist()
                            raw_seq_len = len(beam_raw_sequence_list)

                            if not beam_raw_sequence_list:
                                beam_metadata_map[beam_id] = (0, raw_seq_len, instance_tuple, original_loc_val, [], True) 
                                continue

                            initial_subproblems = _split_sequence_by_zeros(beam_raw_sequence_list)
                            if not initial_subproblems:
                                beam_metadata_map[beam_id] = (0, raw_seq_len, instance_tuple, original_loc_val, [], True)
                                continue
                            
                            merged_node_lists = merge_subproblems_by_centroid_fixed_size(
                                initial_subproblems, 
                                original_loc_val, 
                                original_depot, 
                                problem_size_for_dynamic_target=validation_problem_size, # Added
                                merge_num=args.merge_num, 
                                target_node_count=args.adaptive_merge_target_size # Changed from solver_native_size
                            )
                            if not merged_node_lists:
                                beam_metadata_map[beam_id] = (0, raw_seq_len, instance_tuple, original_loc_val, [], True)
                                continue

                            current_beam_subproblem_instances = []
                            num_original_nodes = len(original_loc_val)
                            valid_merged_node_lists_for_beam = []

                            for node_indices in merged_node_lists:
                                valid_indices = [idx for idx in node_indices if 1 <= idx <= num_original_nodes]
                                if not valid_indices:
                                    # This specific merged list is invalid, but doesn't mean the whole beam is yet
                                    continue 
                                sub_instance = create_subproblem_instance(instance_tuple, current_problem_type, valid_indices)
                                if sub_instance:
                                    current_beam_subproblem_instances.append(sub_instance)
                                    valid_merged_node_lists_for_beam.append(valid_indices) # Store valid node lists for stats
                                else:
                                    # Failure to create one sub_instance for a valid merged_list might mean penalty for whole beam
                                    logger.warning(f"Failed to create subproblem for beam {beam_id}, merged list {valid_indices}. This beam may be penalized.")
                                    # For now, let's assume if one create fails, the beam might be problematic.
                                    # We can decide to penalize the whole beam if current_beam_subproblem_instances ends up empty.

                            if not current_beam_subproblem_instances: # If all merged_lists failed or were empty
                                beam_metadata_map[beam_id] = (0, raw_seq_len, instance_tuple, original_loc_val, [], True)
                                continue
                            
                            # This beam is valid and has subproblems
                            start_index_in_global_list = len(all_subproblems_global)
                            all_subproblems_global.extend(current_beam_subproblem_instances)
                            beam_metadata_map[beam_id] = (len(current_beam_subproblem_instances), raw_seq_len, 
                                                          instance_tuple, original_loc_val, 
                                                          valid_merged_node_lists_for_beam, False, 
                                                          start_index_in_global_list)
                    
                    # --- Solve all collected subproblems in one batch --- 
                    flat_solver_results = None
                    if all_subproblems_global:
                        try:
                            padded_global_subproblems, target_pad_size_global = pad_subproblem_batch(all_subproblems_global, current_problem_type)
                            if not padded_global_subproblems:
                                 raise ValueError("Failed to pad global subproblem batch")
                            tensor_data_global = prepare_batch_tensor_data(padded_global_subproblems, current_problem_type, device)
                            if not tensor_data_global:
                                raise ValueError("Failed to prepare tensor data for global subproblem batch")
                            
                            # We pass all_subproblems_global as original_instance_tuples to solve_vrp_batch
                            # because each is an independent sub-VRP problem instance.
                            flat_solver_results = solve_vrp_batch(
                                solver_model=fixed_solver_model,
                                solver_env_class=CurrentSolverEnvClass,
                                original_instance_tuples=all_subproblems_global, 
                                padded_batch_data=tensor_data_global,
                                padded_problem_size=target_pad_size_global,
                                problem_type=current_problem_type, # Subproblems inherit type
                                device=device,
                                aug_factor=args.solver_aug_factor
                            )
                            if not flat_solver_results or len(flat_solver_results) != len(all_subproblems_global):
                                logger.error(f"Global subproblem solver results mismatch. Expected {len(all_subproblems_global)}, got {len(flat_solver_results) if flat_solver_results else 0}. Penalizing relevant beams.")
                                flat_solver_results = None # Indicate failure
                        except Exception as e:
                            logger.error(f"Error solving global subproblem batch (N={validation_problem_size}): {e}. Penalizing relevant beams.")
                            flat_solver_results = None # Indicate failure
                    
                    # --- Aggregate results for each beam, then find best beam per instance --- 
                    for i in range(current_actual_val_batch_s): # Iterate original instances in batch again
                        best_reward_for_instance = -float('inf')
                        best_seq_len_for_instance = 0
                        best_beam_final_subproblem_node_lists = [] # For stats of the best beam

                        for beam_idx in range(args.beam_size):
                            beam_id = (i, beam_idx)
                            meta = beam_metadata_map.get(beam_id)
                            
                            if not meta:
                                # Should not happen if populated correctly, but as a safeguard
                                current_beam_reward = -args.reward_fail_penalty
                                current_beam_seq_len = 0 # Or some default for seq len if beam was invalid from start
                                current_beam_merged_node_lists = []
                            else:
                                num_subproblems, raw_seq_len, _, _, merged_node_lists, penalized_early, *rest = meta
                                current_beam_seq_len = raw_seq_len
                                current_beam_merged_node_lists = merged_node_lists

                                if penalized_early or num_subproblems == 0:
                                    current_beam_reward = -args.reward_fail_penalty
                                elif flat_solver_results is None: # Global solver failed
                                    current_beam_reward = -args.reward_fail_penalty * num_subproblems # Penalize based on expected work
                                else:
                                    start_idx_global = rest[0]
                                    beam_total_cost = 0
                                    sub_results_for_beam = flat_solver_results[start_idx_global : start_idx_global + num_subproblems]
                                    
                                    if len(sub_results_for_beam) != num_subproblems: # Should be caught by earlier check too
                                        beam_total_cost = args.reward_fail_penalty * num_subproblems
                                    else:
                                        for cost, _ in sub_results_for_beam:
                                            if cost == float('inf') or cost is None or math.isnan(cost):
                                                beam_total_cost += args.reward_fail_penalty
                                            else:
                                                beam_total_cost += cost
                                    current_beam_reward = -beam_total_cost

                            if current_beam_reward > best_reward_for_instance:
                                best_reward_for_instance = current_beam_reward
                                best_seq_len_for_instance = current_beam_seq_len
                                best_beam_final_subproblem_node_lists = current_beam_merged_node_lists
                        
                        # Update meters for this instance based on its best beam
                        partitioned_reward_meter.update(best_reward_for_instance)
                        seq_len_meter.update(best_seq_len_for_instance)
                        
                        num_subproblems_best_beam = len(best_beam_final_subproblem_node_lists)
                        best_beam_subproblem_counts.append(num_subproblems_best_beam)
                        if best_beam_final_subproblem_node_lists:
                            for sub_nodes in best_beam_final_subproblem_node_lists:
                                best_beam_subproblem_node_counts.append(len(sub_nodes))
                    
                    pbar_val.set_postfix(part_reward=f"{partitioned_reward_meter.avg:.2f}", base_reward=f"{avg_baseline_reward:.2f}")
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx} (N={validation_problem_size}): {e}")
                    continue # Skip this batch and move to next
    else: # No batches to run for partitioned rewards
        logger.info(f"No partitioned reward batches to run for {current_problem_type} N={validation_problem_size}.")


    # --- Calculate Final Statistics ---
    avg_partitioned_reward = partitioned_reward_meter.avg if partitioned_reward_meter.count > 0 else (-args.reward_fail_penalty if num_val_instances > 0 else 0)

    partitioned_cost = -avg_partitioned_reward
    # Ensure avg_baseline_reward reflects penalty if no instances ran, matching avg_partitioned_reward logic
    baseline_cost = -avg_baseline_reward 

    # Calculate Improvement Gap
    gap_percent = 0.0
    if abs(baseline_cost) > 1e-9 : # Check if baseline_cost is not effectively zero
        if baseline_cost > 0: # Normal case where baseline cost is positive (higher is worse)
             gap_percent = ((baseline_cost - partitioned_cost) / baseline_cost) * 100
        elif baseline_cost < 0: # Baseline cost is negative (e.g. all penalties, so -(-value) = value)
            # If both are "rewards" (negative costs), and baseline_cost is less negative (better)
            # Example: baseline_cost = -10 (reward 10), partitioned_cost = -12 (reward 12) -> improvement
            # ((-10) - (-12)) / |-10| = 2 / 10 = 20% (Improvement in terms of cost reduction)
            # If baseline_cost = -10 (reward 10), partitioned_cost = -8 (reward 8) -> degradation
            # ((-10) - (-8)) / |-10| = -2 / 10 = -20%
             gap_percent = ((baseline_cost - partitioned_cost) / abs(baseline_cost)) * 100
        # If baseline_cost is exactly zero (e.g. perfect solution or no problem), gap is tricky.
    elif partitioned_cost < 0 : # Baseline is zero, partitioned is better (negative cost/positive reward)
        gap_percent = float('inf') # Infinite improvement
    elif partitioned_cost > 0: # Baseline is zero, partitioned is worse
        gap_percent = float('-inf') # Infinite degradation
    else: # Both are zero
        gap_percent = 0.0


    # Calculate Subproblem Count Stats from best beams
    avg_subproblems_per_instance = np.mean(best_beam_subproblem_counts) if best_beam_subproblem_counts else 0
    avg_nodes_per_subproblem = np.mean(best_beam_subproblem_node_counts) if best_beam_subproblem_node_counts else 0
    subproblem_count_dist = Counter(best_beam_subproblem_counts) # Use new stats
    subproblem_node_count_dist = Counter(best_beam_subproblem_node_counts) # Use new stats

    separator_len = 60
    # Determine log prefix based on whether it's standard or generalization validation
    log_prefix = "Gen. Val." if validation_problem_size != args.problem_size else "Validation"
    logging.info("-" * separator_len)
    logging.info(f"--------------- {log_prefix} Summary (Epoch {current_epoch}, {current_problem_type} N={validation_problem_size}) ---------------")
    logging.info(f"  Avg Partitioned Reward : {avg_partitioned_reward:.4f} (Cost: {partitioned_cost:.4f})")
    logging.info(f"  Avg Baseline Reward    : {avg_baseline_reward:.4f} (Cost: {baseline_cost:.4f})")
    logging.info(f"  Improvement Gap (Cost) : {gap_percent:.2f}% (vs Direct N={validation_problem_size} Solve)") # Clarify gap comparison
    logging.info(f"  Avg Sequence Length    : {seq_len_meter.avg:.2f}")
    logging.info(f"  Avg Subproblems/Instance: {avg_subproblems_per_instance:.2f}")
    logging.info(f"  Subproblem Count Dist  : {dict(subproblem_count_dist.most_common())}")
    logging.info(f"  Avg Nodes/Subproblem   : {avg_nodes_per_subproblem:.2f}")
    logging.info(f"  Subproblem Node Dist   : {dict(subproblem_node_count_dist.most_common(15))}")
    logging.info("-" * separator_len)

    # --- Extract values before deleting meters ---
    final_avg_partitioned_reward = partitioned_reward_meter.avg
    final_avg_baseline_reward = baseline_reward_meter.avg
    final_avg_seq_len = seq_len_meter.avg
    # avg_subproblems_per_instance and avg_nodes_per_subproblem are already calculated scalars
    # subproblem_count_dist and subproblem_node_count_dist are already calculated Counters

    # --- Explicitly release memory --- 
    try:
        del partitioned_reward_meter, baseline_reward_meter, seq_len_meter
        del best_beam_subproblem_counts, best_beam_subproblem_node_counts
        del current_validation_data # Delete reference to input data passed
        # Delete intermediate baseline calculation variables if they exist (might be out of scope, but doesn't hurt)
        if 'padded_batch_b' in locals(): del padded_batch_b
        if 'tensor_data_b' in locals(): del tensor_data_b
        if 'solver_results_b' in locals(): del solver_results_b
        # Delete intermediate partitioned reward calculation variables
        if 'val_env' in locals(): del val_env
        if 'padded_val_batch' in locals(): del padded_val_batch
        if 'val_tensor_data' in locals(): del val_tensor_data
        if 'all_batch_beam_sequences_tensor' in locals(): del all_batch_beam_sequences_tensor
        if 'all_subproblems_global' in locals(): del all_subproblems_global
        if 'beam_metadata_map' in locals(): del beam_metadata_map
        if 'padded_global_subproblems' in locals(): del padded_global_subproblems
        if 'tensor_data_global' in locals(): del tensor_data_global
        if 'flat_solver_results' in locals(): del flat_solver_results
        logger.debug(f"Memory cleanup attempted for validation N={validation_problem_size}")
    except NameError: # Handle cases where some variables might not have been defined (e.g., if execution failed early)
        logger.debug(f"Some variables not defined during memory cleanup for validation N={validation_problem_size}")
        pass 
    # Call PyTorch cache empty and Python garbage collector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.debug(f"gc.collect() called after validation N={validation_problem_size}")

    # Return stats for potential aggregation
    return {
        'validation_problem_size': validation_problem_size, # Include size in results
        'partitioned_reward': final_avg_partitioned_reward,
        'baseline_reward': final_avg_baseline_reward,
        'gap_percent': gap_percent,
        'avg_seq_len': final_avg_seq_len,
        'avg_subproblems': avg_subproblems_per_instance, # Already a scalar
        'subproblem_count_dist': subproblem_count_dist, # Already a Counter
        'avg_nodes_per_subproblem': avg_nodes_per_subproblem, # Already a scalar
        'subproblem_node_dist': subproblem_node_count_dist # Already a Counter
    }

# --- Main Training Function ---
def train_implicit_partitioner(args):
    seed_everything(args.seed)
    # --- Device Setup --- 
    if not args.no_cuda and torch.cuda.is_available():
        # Use the specified GPU ID if CUDA is available
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(device) # Explicitly set the current device for this process
        logging.info(f"Using GPU ID: {args.gpu_id}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
        args.gpu_id = None # Set gpu_id to None if using CPU for clarity elsewhere
    # --- End Device Setup ---

    logging.info(f"Using device: {device}") # Log the final device object

    # --- Load Data & Split ---
    train_datasets = {} # Dict: {problem_type: [instance_tuples]}
    val_datasets_standard = {}   # Dict: {problem_type: [instance_tuples]}
    all_env_classes = {} # Dict: {problem_type: EnvClass}

    if args.problem == "Train_ALL":
        problems_to_load = TRAIN_ALL_PROBLEMS
    else:
        problems_to_load = [args.problem] # Load only the specified problem

    logger.info(f"Loading data for problems: {problems_to_load} with size {args.problem_size}")

    for pt in problems_to_load:
        problem_upper = pt.upper()
        logger.info(f"-- Loading {problem_upper} data...")
        try:
            # Get Env Class first
            try:
                EnvClassList = get_env(problem_upper)
                if not EnvClassList: raise ValueError(f"No EnvClass found for {problem_upper}")
                all_env_classes[problem_upper] = EnvClassList[0]
            except Exception as e:
                logger.error(f"Failed to get Environment class for {problem_upper}: {e}. Skipping this problem.")
                continue
            
            # Get Dataset Path
            if problem_upper not in DATASET_PATHS or args.problem_size not in DATASET_PATHS[problem_upper]:
                 logger.warning(f"Dataset path definition not found for {problem_upper} size {args.problem_size} in DATASET_PATHS. Skipping.")
                 continue
            dataset_info = DATASET_PATHS[problem_upper][args.problem_size]
            dataset_path = dataset_info['data']

            # Check relative path as fallback
            if not os.path.exists(dataset_path):
                 script_dir = os.path.dirname(__file__) if "__file__" in locals() else "."
                 relative_path = os.path.join(script_dir, dataset_path)
                 if os.path.exists(relative_path):
                      dataset_path = relative_path
                      logger.info(f"Using relative dataset path: {dataset_path}")
                 else:
                      logger.warning(f"Dataset file not found at {dataset_info['data']} or {relative_path}. Skipping {problem_upper}.")
                      continue

            full_dataset = load_dataset(dataset_path)
            logger.info(f"Loaded {len(full_dataset)} total instances for {problem_upper} from {dataset_path}")

            # Split dataset
            if args.validation_size > 0 and args.validation_size < len(full_dataset):
                val_datasets_standard[problem_upper] = full_dataset[:args.validation_size]
                train_datasets[problem_upper] = full_dataset[args.validation_size:]
                logger.info(f"Using {len(val_datasets_standard[problem_upper])} instances for validation, {len(train_datasets[problem_upper])} for training {problem_upper}.")
            elif args.validation_size >= len(full_dataset):
                 val_datasets_standard[problem_upper] = full_dataset
                 train_datasets[problem_upper] = []
                 logger.warning(f"Validation size ({args.validation_size}) >= dataset size ({len(full_dataset)}) for {problem_upper}. Using all for validation.")
            else:
                train_datasets[problem_upper] = full_dataset
                val_datasets_standard[problem_upper] = []
                logger.info(f"Using all {len(train_datasets[problem_upper])} instances for training {problem_upper} (validation_size <= 0).")

        except KeyError as e:
            logger.error(f"Dataset definition error for {problem_upper}: {e}. Skipping.")
        except FileNotFoundError as e:
            logger.error(f"Dataset file error for {problem_upper}: {e}. Skipping.")
        except Exception as e:
            logger.error(f"Error loading/splitting dataset for {problem_upper}: {e}. Skipping.")

    if not train_datasets and args.epochs > 0:
        logger.error("No training data loaded for any specified problem. Exiting.")
        return
    if not val_datasets_standard:
         logger.warning("No validation data loaded for any specified problem.")

    # Determine the problem type for the fixed solver env class (can still be args.problem if single task)
    # For multi-task, assume the solver can handle the types it sees, use the first available env class? Or maybe CVRPEnv as default?
    # Let's keep it dynamic for now, get it inside reward/baseline functions
    # fixed_solver_env_class = None # Will be determined per instance
    # Get one env class for sanity check, or default if empty
    default_fixed_env_prob = problems_to_load[0] if problems_to_load else "CVRP"
    try:
        fixed_solver_env_class_check = get_env(default_fixed_env_prob)[0]
        logger.info(f"Using FIXED Solver Environment check with: {fixed_solver_env_class_check.__name__} (will adapt per instance)")
    except Exception as e:
        logger.error(f"Failed to get ANY environment class ({default_fixed_env_prob}): {e}. Exiting.")
        return

    # --- Load FIXED Solver Model (Load ONCE) ---
    logging.info(f"Loading FIXED Solver model from: {args.solver_checkpoint}")
    fixed_solver_model_params = DEFAULT_MODEL_PARAMS.copy() # Start with defaults for fixed solver
    solver_checkpoint_path_norm = args.solver_checkpoint.lower().replace('\\', '/')
    match_solver_experts = re.search(r'mvmoe_(\d+)e_', solver_checkpoint_path_norm)
    solver_model_type = "MOE" # Default assumption for fixed solver
    if match_solver_experts:
        try:
            num_solver_experts = int(match_solver_experts.group(1))
            fixed_solver_model_params['num_experts'] = num_solver_experts
            logging.info(f"Inferred FIXED Solver num_experts={num_solver_experts} from path.")
        except (ValueError, IndexError): pass # Keep default if parsing fails
    if '_light_' in solver_checkpoint_path_norm:
        solver_model_type = "MOE_LIGHT"
        logging.info("Inferred FIXED Solver type as MOE_LIGHT from path.")

    # Pass the main problem type (or 'Train_ALL') to load_moe_model's params if it uses it
    fixed_solver_model_params['problem'] = args.problem
    fixed_solver_model = load_moe_model(args.solver_checkpoint, device, model_type=solver_model_type, model_params=fixed_solver_model_params)
    if not fixed_solver_model:
        logging.error("Failed to load FIXED SOLVER model. Exiting.")
        return

    # --- Initialize MODEL TO BE TRAINED ---
    logging.info(f"Initializing model to be trained (Type: {args.model_type})")
    # Gather model params from args
    model_params = {
        "embedding_dim": args.embedding_dim, "sqrt_embedding_dim": args.sqrt_embedding_dim,
        "encoder_layer_num": args.encoder_layer_num, "decoder_layer_num": args.decoder_layer_num,
        "qkv_dim": args.qkv_dim, "head_num": args.head_num, "logit_clipping": args.logit_clipping,
        "ff_hidden_dim": args.ff_hidden_dim,
        "num_experts": args.num_experts if "MOE" in args.model_type else 0, # Use experts based on type
        "eval_type": args.eval_type, # Use arg for training eval type
        "norm": args.norm, "norm_loc": args.norm_loc,
        "expert_loc": args.expert_loc if "MOE" in args.model_type else [],
        "topk": args.topk if "MOE" in args.model_type else 0,
        "routing_level": args.routing_level if "MOE" in args.model_type else None,
        "routing_method": args.routing_method if "MOE" in args.model_type else None,
        "problem": args.problem, # Pass problem type (e.g., Train_ALL) to model if it uses it
        "device": device
    }
    model_to_train = get_model(args.model_type)(**model_params).to(device)
    logging.info(f"Model to train ({args.model_type}) initialized.")
    num_param(model_to_train) # Log parameter count

    # --- Load Checkpoint for Model to be Trained (Optional Start) ---
    if args.model_checkpoint_start and os.path.exists(args.model_checkpoint_start):
        logging.info(f"Loading weights for model_to_train from: {args.model_checkpoint_start}")
        try:
            # Explicitly set weights_only=False
            start_checkpoint = torch.load(args.model_checkpoint_start, map_location=device, weights_only=False)
            model_state_dict = start_checkpoint.get('model_state_dict', start_checkpoint)
            # Load with strict=False allows starting from potentially different architectures/saved states
            missing_keys, unexpected_keys = model_to_train.load_state_dict(model_state_dict, strict=False)
            if unexpected_keys: logging.warning(f"Ignoring unexpected keys loading model_to_train: {unexpected_keys}")
            if missing_keys: logging.warning(f"Missing keys loading model_to_train (will keep initialized): {missing_keys}")
            logging.info("Successfully loaded compatible weights into model_to_train (strict=False).")
        except Exception as e:
            logging.warning(f"Could not load state_dict into model_to_train from {args.model_checkpoint_start}: {e}. Starting from scratch.")
    elif args.model_checkpoint_start:
        logging.warning(f"Starting checkpoint for model_to_train not found: {args.model_checkpoint_start}. Starting from scratch.")

    # --- Initialize Optimizer and Scheduler ---
    optimizer = optim.Adam(model_to_train.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    # --- Initialize EMA Baseline ---
    baseline_reward_ema = {'ema': 0.0, 'count': 0} # DEPRECATED, using greedy rollout baseline
    start_epoch = 1

    # --- Load Training Checkpoint (Resuming) ---
    if args.resume and os.path.exists(args.resume):
        logging.info(f"Resuming training from checkpoint: {args.resume}")
        try:
            # Explicitly set weights_only=False
            resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model_to_train.load_state_dict(resume_checkpoint['model_state_dict']) # strict=True for resume
            optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
            # Check if scheduler state exists before loading
            if 'scheduler_state_dict' in resume_checkpoint:
                 scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
            else:
                 logging.warning("Scheduler state not found in resume checkpoint. Reinitializing scheduler.")
            start_epoch = resume_checkpoint['epoch'] + 1
            # baseline_reward_ema = resume_checkpoint.get('baseline_reward_ema', baseline_reward_ema) # Load baseline # EMA baseline no longer used
            if 'baseline_reward_ema' in resume_checkpoint: # Check if old checkpoint has it
                logging.info("Loaded EMA baseline from checkpoint, but it will not be used (greedy rollout baseline is active).")
            logging.info(f"Resumed training from epoch {start_epoch}")
        except Exception as e:
            logging.error(f"Error loading resume checkpoint {args.resume}: {e}. Starting from epoch 1.")
            start_epoch = 1
            # baseline_reward_ema = {'ema': 0.0, 'count': 0} # Reset baseline # EMA baseline no longer used

    # --- Training Loop ---
    logging.info(f"Starting training from epoch {start_epoch}...")
    time_estimator = TimeEstimator()

    # Prepare environment class for training loop - THIS IS NOW A DICT: all_env_classes
    # try:
    #     TrainEnvClass = get_env(args.problem)[0]
    # except Exception as e:
    #     logger.error(f"Failed to get Environment class {args.problem} for training: {e}")
    #     return
    # Environment parameters for training loop (pomo size is fixed, problem size might vary if data allows)
    env_params_base = {"pomo_size": args.pomo_size} 

    # Determine which problems to sample from during training
    training_problem_types = list(train_datasets.keys())
    if not training_problem_types:
        logger.error("No training data loaded. Cannot start training loop.")
        return
    logger.info(f"Will sample from training problems: {training_problem_types}")

    for epoch in range(start_epoch, args.epochs + 1):
        model_to_train.train() # Set model to train mode
        model_to_train.set_eval_type(args.eval_type) # Ensure sampling/softmax for REINFORCE
        fixed_solver_model.eval() # Fixed solver is always eval

        epoch_loss_meter = AverageMeter() # Avg Reinforce loss (policy gradient part)
        epoch_reward_meter = AverageMeter() # Avg actual reward (from fixed solver)
        epoch_moe_loss_meter = AverageMeter() # Avg MoE aux loss from model_to_train
        time_estimator.reset(epoch)

        # Determine number of batches based on epoch size and batch size
        num_batches = math.ceil(args.epoch_size / args.train_batch_size)
        instance_count_this_epoch = 0
        logging.info(f"Epoch {epoch}: processing ~{args.epoch_size} instances in {num_batches} batches (batch_size={args.train_batch_size})")

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        optimizer.zero_grad() # Zero grad at the start of the epoch
        accum_count = 0

        for batch_idx in pbar:
            # Sample a problem type first, then sample instances for the batch
            # This ensures each batch comes from a single problem type, simplifying env handling
            current_problem_type = random.choice(training_problem_types)
            current_train_data = train_datasets[current_problem_type]
            if not current_train_data:
                 logger.warning(f"No training data for {current_problem_type}, skipping sampling for this batch.")
                 continue # Skip to next batch iteration
                 
            # Sample indices for the batch from the chosen problem type
            batch_size = min(args.train_batch_size, args.epoch_size - instance_count_this_epoch, len(current_train_data))
            if batch_size <= 0: continue # Skip if epoch size met or no data for this type

            batch_indices = random.sample(range(len(current_train_data)), k=batch_size)
            batch_instance_tuples_original = [current_train_data[i] for i in batch_indices]
            
            # --- Apply Data Augmentation ---
            batch_instance_tuples = []
            if args.apply_data_augmentation:
                for inst_tuple in batch_instance_tuples_original:
                    augmented_inst = _augment_vrp_instance(
                        inst_tuple,
                        current_problem_type, # problem_type for this batch
                        VRP_DATA_FORMAT,      # The VRP_DATA_FORMAT dict
                        args.demand_perturb_factor
                        # apply_geometric_aug and apply_demand_aug are True by default in helper
                    )
                    batch_instance_tuples.append(augmented_inst)
            else:
                batch_instance_tuples = batch_instance_tuples_original
            # --- End Data Augmentation ---

            # Store original problem type for each instance if Train_ALL, for reward calculation
            batch_problem_types = [current_problem_type] * batch_size

            try:
                # --- Get correct Env Class for this batch --- 
                try:
                    CurrentTrainEnvClass = all_env_classes[current_problem_type]
                except KeyError:
                    logger.error(f"Missing EnvClass for {current_problem_type} in all_env_classes dict. Skipping batch.")
                    continue
                    
                # --- Environment Setup for Batch --- 
                # Update env_params with current problem size (assuming it's fixed per type for now)
                current_env_params = {**env_params_base, "problem_size": args.problem_size}
                train_env = CurrentTrainEnvClass(**current_env_params)
                
                # Prepare tensor data for the environment batch
                # Use current_problem_type for padding/preparation
                padded_batch_tuples, target_pad_size = pad_subproblem_batch(batch_instance_tuples, current_problem_type)
                if not padded_batch_tuples: raise ValueError(f"Failed to pad training batch for {current_problem_type}")
                batch_tensor_data = prepare_batch_tensor_data(
                    padded_batch_tuples, current_problem_type, device
                )
                if not batch_tensor_data: raise ValueError(f"Failed to prepare tensor data for training batch ({current_problem_type})")

                train_env.load_problems(batch_size=batch_size, problems=batch_tensor_data, aug_factor=1) # No augmentation for generation

                # --- Rollout using model_to_train (Sampling for actual actions) --- 
                original_eval_type = model_to_train.eval_type # Save eval type
                model_to_train.set_eval_type(args.eval_type) # Ensure sampling for REINFORCE

                reset_state, _, _ = train_env.reset()
                model_to_train.pre_forward(reset_state)

                log_prob_list = torch.zeros(size=(batch_size, args.pomo_size, 0), device=device)
                state, _, done = train_env.pre_step()
                while not done:
                    selected, prob = model_to_train(state)
                    state, _, done = train_env.step(selected)
                    # Ensure prob has the right shape (batch, pomo) AND is on the correct device before unsqueezing/concatenating
                    if prob.dim() == 2 and prob.shape == (batch_size, args.pomo_size):
                         # Make sure prob is on the correct device before appending
                         prob_on_device = prob.to(device)
                         log_prob_list = torch.cat((log_prob_list, prob_on_device[:, :, None]), dim=2)
                    elif prob.dim() == 1 and prob.shape[0] == batch_size * args.pomo_size: # Handle potential flattening
                         prob_reshaped = prob.reshape(batch_size, args.pomo_size)
                         # Make sure reshaped prob is on the correct device before appending
                         prob_reshaped_on_device = prob_reshaped.to(device)
                         log_prob_list = torch.cat((log_prob_list, prob_reshaped_on_device[:, :, None]), dim=2)
                    else:
                         # Add device info to the error message for better debugging
                         raise ValueError(f"Unexpected probability shape or device from model: {prob.shape} on {prob.device}, expected ({batch_size}, {args.pomo_size}) on {device}")


                # Check if sequences were generated
                batch_raw_sequences = []
                if hasattr(train_env, 'selected_node_list') and train_env.selected_node_list is not None:
                    # Shape: (batch*aug, pomo, seq_len) -> (batch*1, pomo, seq_len)
                    if train_env.selected_node_list.numel() > 0:
                        sequences_tensor = train_env.selected_node_list.view(batch_size, args.pomo_size, -1)
                        # Extract sequences for each instance in the batch (assuming pomo=1 for now)
                        for i in range(batch_size):
                            # TODO: If pomo > 1, need to handle multiple sequences per instance
                            if args.pomo_size == 1:
                                batch_raw_sequences.append(sequences_tensor[i, 0, :].cpu().tolist())
                            else:
                                # Need logic to handle multiple POMO sequences if used
                                logger.warning("POMO > 1 not fully implemented for reward calculation yet. Using first sequence.")
                                batch_raw_sequences.append(sequences_tensor[i, 0, :].cpu().tolist())
                    else:
                         logger.warning(f"Batch {batch_idx}: env.selected_node_list is empty.")
                else:
                     logger.warning(f"Batch {batch_idx}: env.selected_node_list not found or None.")

                if len(batch_raw_sequences) != batch_size:
                    logger.error(f"Batch {batch_idx}: Mismatch between generated sequences ({len(batch_raw_sequences)}) and batch size ({batch_size}). Skipping batch.")
                    model_to_train.set_eval_type(original_eval_type) # Restore eval type
                    continue

                # --- Calculate Actual Rewards for the Sampled Batch --- 
                batch_rewards_actual_list = []
                # Get the correct solver env class for the current problem type (already have CurrentTrainEnvClass, but reward needs its own based on problem_type)
                
                for i in range(batch_size):
                    # Determine problem type for this specific instance (if Train_ALL)
                    # For now, assumes batch_problem_types[i] is correct
                    instance_problem_type_for_reward = batch_problem_types[i]
                    try:
                        RewardSolverEnvClass = all_env_classes[instance_problem_type_for_reward]
                    except KeyError:
                        logger.error(f"Missing EnvClass for reward calculation ({instance_problem_type_for_reward}). Applying penalty.")
                        batch_rewards_actual_list.append(-args.reward_fail_penalty)
                        continue

                    try:
                        original_loc_i = batch_instance_tuples[i][1]
                    except (IndexError, TypeError):
                        logger.error(f"Could not extract original_loc for instance {i} in batch (actual reward). Applying penalty.")
                        batch_rewards_actual_list.append(-args.reward_fail_penalty)
                        continue
                    
                    reward_actual, _ = calculate_partition_reward( 
                        batch_raw_sequences[i], batch_instance_tuples[i],
                        original_loc_i,
                        instance_problem_type_for_reward, 
                        args.merge_num, fixed_solver_model, RewardSolverEnvClass,
                        problem_size_for_dynamic_target=args.problem_size, # Added (current training problem size)
                        adaptive_merge_target_from_args=args.adaptive_merge_target_size, # Added
                        solver_aug_factor=args.solver_aug_factor, device=device,
                        fail_penalty=args.reward_fail_penalty
                    )
                    batch_rewards_actual_list.append(reward_actual)
                batch_rewards_actual_tensor = torch.tensor(batch_rewards_actual_list, device=device)

                # --- Greedy Rollout for Baseline ---
                model_to_train.set_eval_type('argmax') # Set to greedy for baseline
                batch_rewards_greedy_list = []

                # Create a new env instance or reset existing for the greedy rollout
                # to ensure state independence from the sampling rollout.
                # Using a new instance is safer if env reset is complex or has side effects.
                with torch.no_grad(): # Baseline calculation should not contribute to model's gradients
                    # Assuming batch_tensor_data is still valid for the same batch_instance_tuples
                    baseline_env = CurrentTrainEnvClass(**current_env_params) # Use same env type and params as training
                    baseline_env.load_problems(batch_size=batch_size, problems=batch_tensor_data, aug_factor=1)
                    
                    reset_state_greedy, _, _ = baseline_env.reset()
                    model_to_train.pre_forward(reset_state_greedy)

                    greedy_state, _, greedy_done = baseline_env.pre_step()
                    # log_prob_list for greedy rollout is not needed
                    while not greedy_done:
                        selected_greedy, _ = model_to_train(greedy_state) # prob from greedy rollout is ignored
                        greedy_state, _, greedy_done = baseline_env.step(selected_greedy)
                    
                    batch_raw_sequences_greedy = []
                    if hasattr(baseline_env, 'selected_node_list') and baseline_env.selected_node_list is not None:
                        if baseline_env.selected_node_list.numel() > 0:
                            greedy_sequences_tensor = baseline_env.selected_node_list.view(batch_size, args.pomo_size, -1)
                            for i in range(batch_size):
                                # POMO size is 1 for greedy baseline generation here
                                batch_raw_sequences_greedy.append(greedy_sequences_tensor[i, 0, :].cpu().tolist()) 
                        else: logger.warning(f"Batch {batch_idx} (Greedy Baseline): baseline_env.selected_node_list is empty.")
                    else: logger.warning(f"Batch {batch_idx} (Greedy Baseline): baseline_env.selected_node_list not found or None.")

                    if len(batch_raw_sequences_greedy) != batch_size:
                        logger.error(f"Batch {batch_idx} (Greedy Baseline): Mismatch in generated greedy sequences ({len(batch_raw_sequences_greedy)}) and batch size ({batch_size}). Using penalty for baseline.")
                        batch_rewards_greedy_list = [-args.reward_fail_penalty] * batch_size
                    else:
                        for i in range(batch_size):
                            instance_problem_type_for_reward = batch_problem_types[i]
                            try:
                                RewardSolverEnvClass = all_env_classes[instance_problem_type_for_reward]
                            except KeyError:
                                batch_rewards_greedy_list.append(-args.reward_fail_penalty)
                                continue
                            try:
                                original_loc_i = batch_instance_tuples[i][1]
                            except (IndexError, TypeError):
                                batch_rewards_greedy_list.append(-args.reward_fail_penalty)
                                continue

                            reward_greedy, _ = calculate_partition_reward(
                                batch_raw_sequences_greedy[i], batch_instance_tuples[i],
                                original_loc_i,
                                instance_problem_type_for_reward,
                                args.merge_num, fixed_solver_model, RewardSolverEnvClass,
                                problem_size_for_dynamic_target=args.problem_size, # Added
                                adaptive_merge_target_from_args=args.adaptive_merge_target_size, # Added
                                solver_aug_factor=args.solver_aug_factor, device=device, 
                                fail_penalty=args.reward_fail_penalty
                            )
                            batch_rewards_greedy_list.append(reward_greedy)
                
                batch_rewards_greedy_tensor = torch.tensor(batch_rewards_greedy_list, device=device)
                
                # --- Restore original eval_type for the model ---
                model_to_train.set_eval_type(original_eval_type)


                # --- Update Baselines and Calculate Loss ---
                # baseline_state = baseline_reward_ema # Deprecated
                # current_baseline = baseline_state['ema'] if baseline_state['count'] > 0 else 0.0 # Deprecated
                # # Update EMA using the mean reward of the batch # Deprecated
                # batch_mean_reward = batch_rewards_tensor.mean().item() # Deprecated
                # new_ema = args.ema_beta * current_baseline + (1 - args.ema_beta) * batch_mean_reward # Deprecated
                # baseline_state['ema'] = new_ema # Deprecated
                # baseline_state['count'] += batch_size # Deprecated

                # Expand baseline and reward for pomo dimension if needed (assuming pomo=1 for sampling rollout)
                expanded_actual_rewards = batch_rewards_actual_tensor.unsqueeze(1).expand(-1, args.pomo_size) # (batch, pomo)
                # Greedy baseline is per instance, so it also needs to be expanded for the pomo dimension of the sampled actions
                expanded_greedy_baseline = batch_rewards_greedy_tensor.unsqueeze(1).expand_as(expanded_actual_rewards) # (batch, pomo)


                advantage = expanded_actual_rewards - expanded_greedy_baseline
                
                # Check if log_prob_list is empty (can happen if rollout failed early)
                if log_prob_list.numel() == 0:
                     logger.warning(f"Batch {batch_idx}: log_prob_list is empty. Skipping loss calculation.")
                     total_log_prob = torch.zeros(batch_size, args.pomo_size, device=device) # Avoid NaN
                     reinforce_loss = torch.tensor(0.0, device=device) # No loss if no actions
                else:
                     total_log_prob = log_prob_list.log().sum(dim=2) # (batch, pomo)
                     reinforce_loss = -(total_log_prob * advantage.detach()).mean() # Average over batch and pomo


                # Add MoE aux loss from the model being trained
                moe_aux_loss = getattr(model_to_train, 'aux_loss', 0.0)
                if isinstance(moe_aux_loss, torch.Tensor) and moe_aux_loss.numel() > 1:
                    moe_aux_loss = moe_aux_loss.mean() # Use mean if it's per-expert/layer loss
                elif not isinstance(moe_aux_loss, torch.Tensor):
                     moe_aux_loss = torch.tensor(moe_aux_loss, device=device)

                total_loss = reinforce_loss + moe_aux_loss * args.moe_loss_coef

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                     logger.error(f"NaN/Inf loss detected for batch {batch_idx}. Skipping update.")
                     # Don't zero grad here, let accumulation handle it or reset at start of loop
                     continue

                loss_for_backward = total_loss / args.grad_accum_steps

                # --- Backward and Accumulate ---
                loss_for_backward.backward()
                epoch_loss_meter.update(reinforce_loss.item(), batch_size)
                epoch_reward_meter.update(batch_rewards_actual_tensor.mean().item(), batch_size) # Log actual rewards
                epoch_moe_loss_meter.update(moe_aux_loss.item(), batch_size)
                accum_count += 1

                # --- Gradient Update Step ---
                if accum_count >= args.grad_accum_steps:
                    if args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model_to_train.parameters(), max_norm=args.grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad() # Zero grad AFTER stepping
                    accum_count = 0 # Reset accumulation counter

                pbar.set_postfix(
                    loss=f"{epoch_loss_meter.avg:.3f}",
                    reward=f"{epoch_reward_meter.avg:.2f}",
                    baseline=f"{batch_rewards_greedy_tensor.mean().item():.2f}", # Show mean greedy baseline
                    moe_loss=f"{epoch_moe_loss_meter.avg:.4f}"
                )
                instance_count_this_epoch += batch_size

            except Exception as batch_e:
                 logger.exception(f"Error processing batch {batch_idx}: {batch_e}")
                 # If an error occurs mid-batch, ensure gradients are cleared before the next batch
                 optimizer.zero_grad()
                 accum_count = 0


        # --- End Epoch ---
        # Perform final optimizer step if accumulated gradients remain
        if accum_count > 0:
            if args.grad_clip_norm > 0:
                 torch.nn.utils.clip_grad_norm_(model_to_train.parameters(), max_norm=args.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            logger.info("Performed final optimizer step for remaining accumulated gradients.")


        scheduler.step() # Step the scheduler at the end of the epoch

        elapsed_str, remain_str = time_estimator.get_est_string(epoch, args.epochs)
        logging.info(f"Epoch {epoch} Summary: Avg Loss={epoch_loss_meter.avg:.4f}, Avg Reward={epoch_reward_meter.avg:.2f}, Avg MoE Loss={epoch_moe_loss_meter.avg:.4f}")
        logging.info(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        logging.info(f"Time Est.: Elapsed[{elapsed_str}], Remain[{remain_str}]")

        # --- Explicitly flush the file handler buffer after epoch summary ---
        file_handler.flush() 

        # --- Standard Validation (on args.problem_size) ---
        if epoch % args.validation_interval == 0 or epoch == args.epochs or epoch == 1:
             all_standard_validation_results = {}
             problems_to_validate = TRAIN_ALL_PROBLEMS if args.problem == "Train_ALL" else [args.problem]
             logger.info(f"--- Starting Standard Validation (N={args.problem_size}) ---")
             for pt_val in problems_to_validate:
                  # Explicitly check if key exists first
                  if pt_val not in val_datasets_standard:
                       logger.info(f"No standard validation data dictionary entry for {pt_val}. Skipping.")
                       continue
                  # Now check if the data list for this key is empty
                  current_pt_val_data = val_datasets_standard[pt_val]
                  if not current_pt_val_data:
                       logger.info(f"Standard validation data list is empty for {pt_val}. Skipping.")
                       continue
                      
                  # Check for environment class (can keep this check)
                  if pt_val not in all_env_classes:
                       logger.warning(f"No env class for {pt_val}. Skipping std validation.")
                       continue
                       
                  # Call validation function with args.problem_size using the correct data
                  validation_stats = _run_validation_for_problem(
                      model_to_train, fixed_solver_model, all_env_classes,
                      current_pt_val_data, args, device, epoch, pt_val, # Use validated current_pt_val_data
                      validation_problem_size=args.problem_size # Specify standard size
                  )
                  if validation_stats: all_standard_validation_results[pt_val] = validation_stats
             # --- Aggregate and Log Standard Validation ---
             if all_standard_validation_results:
                  # Use nanmean to handle potential NaN gaps
                  avg_gap_std = np.nanmean([stats['gap_percent'] for stats in all_standard_validation_results.values() if stats])
                  avg_reward_std = np.nanmean([stats['partitioned_reward'] for stats in all_standard_validation_results.values() if stats])
                  logger.warning(f"=== Overall Standard Validation Summary (Epoch {epoch}, N={args.problem_size}) ===")
                  logger.warning(f"  Avg Gap across validated problems : {avg_gap_std:.2f}%")
                  logger.warning(f"  Avg Reward across validated problems: {avg_reward_std:.4f}")
                  logger.warning(f"===================================================")
             logger.info(f"--- Finished Standard Validation ---") # Added footer
             file_handler.flush()
             # --- Added Memory Release after Standard Validation ---
             if torch.cuda.is_available():
                 torch.cuda.empty_cache()
             gc.collect()
             logger.debug("Memory release attempted after standard validation.")
             
        # --- Generalization Validation (on larger sizes) ---
        if args.generalization_validation_interval > 0 and (epoch % args.generalization_validation_interval == 0 or epoch == args.epochs or epoch == 1):
            logger.info(f"--- Starting Generalization Validation (Epoch {epoch}) ---")
            # Define potential generalization sizes (Removed 5000)
            all_generalization_sizes = args.generalization_sizes if args.generalization_sizes else [500, 1000, 2000]
            # Dynamically select sizes strictly larger than the training size
            generalization_sizes_to_run = [s for s in all_generalization_sizes if s > args.problem_size]
            
            if not generalization_sizes_to_run:
                 logger.info(f"No generalization sizes larger than training size ({args.problem_size}) defined. Skipping generalization validation.")
            else:
                logger.info(f"Selected generalization sizes to run: {generalization_sizes_to_run}")
                all_gen_validation_results = {} # Store results per size and problem
                problems_to_validate_gen = TRAIN_ALL_PROBLEMS if args.problem == "Train_ALL" else [args.problem]
    
                # Loop through the dynamically selected sizes
                for gen_size in generalization_sizes_to_run:
                     logger.info(f"--- Generalization Validation for N={gen_size} ---")
                     all_gen_validation_results[gen_size] = {}
                     # Track if any validation was run for this size
                     validation_run_for_size = False 
                     for pt_val in problems_to_validate_gen:
                          # Load limited samples for this size/problem
                          gen_val_data = []
                          try:
                              if pt_val not in DATASET_PATHS or gen_size not in DATASET_PATHS[pt_val]:
                                  logger.debug(f"No dataset defined for {pt_val} N={gen_size}. Skipping.") # Use debug level
                                  continue
                              dataset_info = DATASET_PATHS[pt_val][gen_size]
                              dataset_path = dataset_info['data']
                              if not os.path.exists(dataset_path):
                                   script_dir = os.path.dirname(__file__) if "__file__" in locals() else "."
                                   relative_path = os.path.join(script_dir, dataset_path)
                                   if os.path.exists(relative_path): dataset_path = relative_path
                                   else: 
                                        logger.debug(f"Dataset file not found for {pt_val} N={gen_size}. Skipping.") # Use debug level
                                        continue
                          
                              full_gen_dataset = load_dataset(dataset_path, disable_print=True) # Disable print for cleaner logs
                              num_samples = min(args.generalization_val_samples, len(full_gen_dataset))
                              if num_samples <= 0: 
                                   logger.debug(f"No samples requested/available for {pt_val} N={gen_size}. Skipping.") # Use debug level
                                   continue
                              gen_val_data = full_gen_dataset[:num_samples]
                              logger.info(f"Loaded {len(gen_val_data)} instances for {pt_val} N={gen_size} generalization validation.")
                          except Exception as e:
                              logger.error(f"Error loading generalization data for {pt_val} N={gen_size}: {e}. Skipping.")
                              continue
                          
                          if not gen_val_data: continue # Skip if no data loaded

                          if pt_val not in all_env_classes:
                               logger.warning(f"No env class for {pt_val}. Skipping gen validation for N={gen_size}.")
                               continue

                          # Call validation function with gen_size
                          gen_stats = _run_validation_for_problem(
                              model_to_train, fixed_solver_model, all_env_classes,
                              gen_val_data, args, device, epoch, pt_val,
                              validation_problem_size=gen_size # Specify generalization size
                          )
                          if gen_stats: 
                               all_gen_validation_results[gen_size][pt_val] = gen_stats
                               validation_run_for_size = True # Mark that validation ran

                     # --- Aggregate and Log Generalization Validation for this size ---
                     if validation_run_for_size and all_gen_validation_results[gen_size]:
                          # Use nanmean to handle potential NaN gaps
                          avg_gap_gen = np.nanmean([stats['gap_percent'] for stats in all_gen_validation_results[gen_size].values() if stats])
                          avg_reward_gen = np.nanmean([stats['partitioned_reward'] for stats in all_gen_validation_results[gen_size].values() if stats])
                          logger.warning(f"=== Overall Generalization Validation Summary (Epoch {epoch}, N={gen_size}) ===")
                          logger.warning(f"  Avg Gap across validated problems : {avg_gap_gen:.2f}%")
                          logger.warning(f"  Avg Reward across validated problems: {avg_reward_gen:.4f}")
                          logger.warning(f"======================================================")
                     elif validation_run_for_size: # Validation ran but no stats collected (errors maybe)
                          logger.warning(f"Generalization validation ran for N={gen_size} but no results collected.")
                     else: # No validation ran for this size (due to missing data/envs)
                          logger.info(f"No generalization validation performed for N={gen_size}.")
                 # End loop over gen_size
                 
                logger.info(f"--- Finished Generalization Validation ---") # Added footer
                file_handler.flush() 
                
            # --- Added Memory Release after Generalization Validation block --- 
            # This block executes only if the generalization validation condition was met
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.debug("Memory release attempted after generalization validation block.")


        # --- Save Checkpoint ---
        if epoch % args.model_save_interval == 0 or epoch == args.epochs:
            logging.info(f"--- Saving Checkpoint for Epoch {epoch} --- ")
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model_to_train.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                # 'baseline_reward_ema': baseline_reward_ema, # Save EMA state # No longer saving EMA baseline
                'args': vars(args) # Save args as dict
            }
            # Use run-specific checkpoint directory and simplified epoch name
            save_path = os.path.join(args.checkpoint_dir, f"{epoch}.pt") 
            try:
                torch.save(checkpoint_data, save_path)
                logging.info(f"Checkpoint saved to {save_path}")
            except Exception as e:
                logging.error(f"Error saving checkpoint: {e}")

    logging.info("Training finished.")

# --- Data Augmentation Helper Functions ---
def _transform_coords(coords, choice):
    """Applies a chosen geometric transformation to a list of coordinates."""
    # coords: list of [x, y]
    # choice: 0-7 for 8 symmetries (0: identity, 1-3: rotations, 4-7: flips)
    transformed = []
    for x, y in coords:
        if choice == 0: # Identity
            nx, ny = x, y
        elif choice == 1: # Rotate 90 deg clockwise around (0.5, 0.5) -> (y, 1-x) assuming coords in [0,1]
            nx, ny = y, 1 - x
        elif choice == 2: # Rotate 180 deg -> (1-x, 1-y)
            nx, ny = 1 - x, 1 - y
        elif choice == 3: # Rotate 270 deg clockwise -> (1-y, x)
            nx, ny = 1 - y, x
        elif choice == 4: # Flip X (across y=0.5 line, if centered at 0.5) -> (x, 1-y)
            nx, ny = x, 1 - y
        elif choice == 5: # Flip Y (across x=0.5 line, if centered at 0.5) -> (1-x, y)
            nx, ny = 1 - x, y
        elif choice == 6: # Flip Diagonal (y=x) -> (y,x)
            nx, ny = y, x
        elif choice == 7: # Flip Anti-Diagonal (y=1-x) -> (1-y, 1-x)
            nx, ny = 1 - y, 1 - x
        else: # Should not happen, defaults to identity
            nx, ny = x, y
        transformed.append([nx, ny])
    return transformed

def _augment_vrp_instance(instance_tuple, problem_type,
                         vrp_data_format_dict, # Pass VRP_DATA_FORMAT
                         demand_perturb_factor=0.1,
                         apply_geometric_aug=True,
                         apply_demand_aug=True):
    """Applies geometric and/or demand augmentation to a single VRP instance tuple."""
    if problem_type not in vrp_data_format_dict:
        logger.warning(f"Problem type {problem_type} not in VRP_DATA_FORMAT. Skipping augmentation for this instance.")
        return instance_tuple

    format_keys = vrp_data_format_dict[problem_type]
    
    # Ensure instance_tuple has enough elements for format_keys
    if len(instance_tuple) < len(format_keys):
        logger.error(f"Instance tuple length {len(instance_tuple)} is less than expected {len(format_keys)} for {problem_type}. Cannot augment.")
        return instance_tuple
        
    instance_dict = {key: val for key, val in zip(format_keys, instance_tuple)}

    # 1. Geometric Augmentation
    if apply_geometric_aug:
        aug_choice = random.randint(0, 7) # 8 symmetries (0 is identity)
        if aug_choice > 0: # Only apply if not identity, to avoid unnecessary list comprehensions
            if "depot_xy" in instance_dict and isinstance(instance_dict["depot_xy"], list):
                instance_dict["depot_xy"] = _transform_coords(instance_dict["depot_xy"], aug_choice)
            if "node_xy" in instance_dict and isinstance(instance_dict["node_xy"], list):
                instance_dict["node_xy"] = _transform_coords(instance_dict["node_xy"], aug_choice)

    # 2. Demand Augmentation
    if apply_demand_aug and "demand" in instance_dict and isinstance(instance_dict["demand"], list):
        original_demands = instance_dict["demand"]
        perturbed_demands = []
        for d_val in original_demands:
            # Ensure demand is a number before operating
            if not isinstance(d_val, (int, float)):
                logger.warning(f"Non-numeric demand value encountered: {d_val}. Skipping demand perturbation for this value.")
                perturbed_demands.append(d_val)
                continue

            # Assuming demands are typically non-negative for partitioning purposes.
            # If backhauls are represented by negative demands and this needs preservation,
            # perturbation logic would need to be sign-aware (e.g., perturb abs value).
            # For now, simple perturbation and ensuring non-negativity.
            perturb = random.uniform(-demand_perturb_factor, demand_perturb_factor)
            new_demand = d_val * (1 + perturb)
            perturbed_demands.append(max(0, new_demand)) # Ensure non-negative demand after perturbation
        instance_dict["demand"] = perturbed_demands
        # Capacity is typically not changed with demand augmentation to vary load factor.

    # Reconstruct the tuple in the correct order
    augmented_instance_list = []
    for key in format_keys:
        if key in instance_dict:
            augmented_instance_list.append(instance_dict[key])
        else:
            # This case should ideally not happen if VRP_DATA_FORMAT is consistent
            # and instance_tuple matches it. Add placeholder or error.
            logger.error(f"Key '{key}' from VRP_DATA_FORMAT not found in instance_dict during augmentation. This indicates a mismatch.")
            # Fallback to original data for this key if possible or handle error
            # For safety, if a key is missing, it might be better to return original tuple or raise error
            # Returning original tuple if a key is missing during reconstruction
            # Find original index of key
            try:
                original_key_index = format_keys.index(key)
                if original_key_index < len(instance_tuple):
                     augmented_instance_list.append(instance_tuple[original_key_index])
                else: # Should not happen if lengths matched initially
                     logger.error(f"Cannot find fallback for missing key '{key}' in original tuple.")
                     return instance_tuple # Safety return
            except ValueError:
                 logger.error(f"Key '{key}' defined in VRP_DATA_FORMAT not found in its own keys list. Skipping augmentation.")
                 return instance_tuple # Safety return

    return tuple(augmented_instance_list)
# --- End Data Augmentation Helper Functions --- 


# --- Main Execution ---
if __name__ == '__main__':
    args = parse_arguments()

    # --- Set default generalization_validation_interval if not provided ---
    if args.generalization_validation_interval is None:
        args.generalization_validation_interval = args.validation_interval
        print(f"Defaulting generalization_validation_interval to validation_interval: {args.generalization_validation_interval}")

    # --- Set default validate_batch_size if not provided ---
    if args.validate_batch_size is None:
        args.validate_batch_size = args.train_batch_size
        # We will log this using the logger once it's set up.
        # print(f"Defaulting validate_batch_size to train_batch_size: {args.validate_batch_size}")

    # --- Setup Logging & Directories ---
    log_dir = args.log_dir
    checkpoint_dir = args.checkpoint_dir
    # Create unique run directories
    run_timestamp = datetime.now(pytz.timezone("Asia/Singapore")).strftime("%Y%m%d_%H%M%S")
    # Make run_id slightly more informative if possible
    model_type_short = args.model_type.replace('_LIGHT','Lt').replace('_','')
    # Include num_experts in run_id
    run_id = f"{model_type_short}_{args.num_experts}e_{args.problem}_n{args.problem_size}_{run_timestamp}"
    run_log_dir = os.path.join(log_dir, run_id)
    run_checkpoint_dir = os.path.join(checkpoint_dir, run_id)

    try:
        if not os.path.exists(run_log_dir): os.makedirs(run_log_dir)
        if not os.path.exists(run_checkpoint_dir): os.makedirs(run_checkpoint_dir)
        # Update args to use run-specific dirs for saving checkpoints later
        args.log_dir = run_log_dir
        args.checkpoint_dir = run_checkpoint_dir
    except OSError as e:
        print(f"Error creating directories: {e}. Check permissions or path.")
        exit(1)

    # --- Configure Logging ---
    log_file_path = os.path.join(run_log_dir, 'training.log')
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s') # Use the same format

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # Set the minimum level for the root logger

    # Clear existing handlers (important to avoid duplicates or interference)
    root_logger.handlers.clear()

    # File Handler (always logs INFO and above)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # Console Handler (level depends on verbosity)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    if args.verbose_log:
        console_handler.setLevel(logging.INFO)
    else:
        # Only show WARNING, ERROR, CRITICAL on console if not verbose
        console_handler.setLevel(logging.WARNING) 
    root_logger.addHandler(console_handler)

    # Now, get the specific logger for this module - it will inherit handlers and levels
    logger = logging.getLogger(__name__) 
    # No need to add handlers directly to 'logger' if root logger is configured correctly

    logger.info(f"Run Log Directory: {run_log_dir}")
    logger.info(f"Run Checkpoint Directory: {run_checkpoint_dir}")
    # Use pprint for cleaner arg printing
    import pprint
    logger.info(f"Arguments: {pprint.pformat(vars(args))}")
    # Log the defaulted validate_batch_size
    if 'validate_batch_size' in vars(args) and args.validate_batch_size == args.train_batch_size and args.train_batch_size is not None:
         logger.info(f"validate_batch_size defaulted to train_batch_size: {args.validate_batch_size}")

    # Print basic start message to console regardless of verbose setting
    # Use print directly for unconditional console output
    print(f"\n>>> Starting training run: {run_id} <<<") 
    print(f"    Logs: {log_file_path}")
    print(f"    Checkpoints: {run_checkpoint_dir}")
    print(f"    To see detailed logs in console, use --verbose_log\n")


    # --- Start Training ---
    try:
        train_implicit_partitioner(args)
    finally:
        # Print basic end message
        print(f"\n>>> Finished training run: {run_id} <<<") 
