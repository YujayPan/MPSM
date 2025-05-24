import argparse
import pickle
import os
import sys
import matplotlib.pyplot as plt # Added for subplots
import numpy as np # Added for np.isnan if score is NaN
import csv # Added for reading summary results

# Adjust system path to include the parent directory for utils and other modules
# This is often necessary when running scripts from a subdirectory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from cvrp_lib_integration.cvrp_lib_parser import parse_vrp_file, to_internal_tuple
    from data_visualize import print_vrp_instance_info, visualize_solution
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure that cvrp_lib_parser.py and data_visualize.py are accessible,")
    print("and that the script is run from a context where 'utils' can be found if imported by them.")
    sys.exit(1)

# Define the methods to be visualized and their expected naming in directories
METHODS_TO_VISUALIZE = [
    'DirectORTools',
    'DirectSolver',
    'PartitionORTools_adaptive',
    'PartitionSolver_adaptive'
]

# --- New: Method Name Mapping for Display ---
METHOD_DISPLAY_NAME_MAP = {
    'DirectSolver': 'Solver',
    'DirectORTools': 'ORTools',
    'PartitionSolver_adaptive': 'MPSM_Solver',
    'PartitionORTools_adaptive': 'MPSM_ORTools'
}
# --- End New ---

# Define the base directory for CVRP-LIB files (can be made a parameter if needed)
CVRPLIB_BASE_DIR = "Vrp-Set-X"
SUMMARY_CSV_FILENAME = "summary_results_multimethod.csv" # Name of the summary CSV

def load_method_metrics_from_csv(batch_run_dir, base_instance_name):
    """
    Loads method metrics (score, total_time_seconds) from the summary CSV file.

    Args:
        batch_run_dir (str): Path to the batch run output directory.
        base_instance_name (str): Base name of the VRP instance (e.g., X-n101-k25).

    Returns:
        dict: A dictionary where keys are method names and values are dicts 
              containing {'score': float, 'time': float}. Returns empty if file not found or error.
    """
    metrics = {}
    # Construct the full path to the CSV file.
    csv_path = os.path.join(batch_run_dir, SUMMARY_CSV_FILENAME)
    
    # Debug prints
    print(f"DEBUG: Attempting to load metrics from CSV: '{csv_path}'")
    print(f"DEBUG: Filtering for base_instance_name: '{base_instance_name}'")
    print(f"DEBUG: Expecting methods for visualization: {METHODS_TO_VISUALIZE}")

    if not os.path.exists(csv_path):
        print(f"Warning: Summary CSV file not found at '{csv_path}'")
        return metrics

    try:
        with open(csv_path, mode='r', newline='', encoding='utf-8') as infile: # Added encoding
            reader = csv.DictReader(infile)
            if reader.fieldnames:
                 print(f"DEBUG: CSV Columns found: {reader.fieldnames}")
            else:
                 print(f"DEBUG: CSV file '{csv_path}' appears to be empty or has no header.")
                 return metrics # Early exit if no columns

            for row_num, row in enumerate(reader):
                # Conditional, less verbose print for each row unless specific matches occur
                # Example: print first 5 rows and all rows matching the base_instance_name
                # if (row_num < 5) or (row.get('problem_name', '').strip() == base_instance_name) :
                #     print(f"DEBUG: Processing CSV row {row_num + 1}: {row}")
                
                problem_name_csv = row.get('problem_name', '').strip() # Add strip
                method_applied_csv = row.get('method_applied', '').strip() # Add strip

                if problem_name_csv == base_instance_name:
                    # This print is useful to confirm instance name match
                    print(f"DEBUG: Matched base_instance_name '{base_instance_name}' in CSV row {row_num +1}. Full row: {row}")
                    
                    if method_applied_csv in METHODS_TO_VISUALIZE:
                        print(f"DEBUG: Matched method_applied '{method_applied_csv}' in METHODS_TO_VISUALIZE.")
                        try:
                            cost_val_str = row.get('cost')
                            time_total_sec_str = row.get('time_total_sec')
                            
                            # print(f"DEBUG: For '{method_applied_csv}', raw cost_str='{cost_val_str}', raw time_str='{time_total_sec_str}'")

                            # Convert to float, handling potential empty strings or None
                            score = float(cost_val_str) if cost_val_str not in [None, ''] else float('nan')
                            time_val = float(time_total_sec_str) if time_total_sec_str not in [None, ''] else float('nan')
                            
                            metrics[method_applied_csv] = {'score': score, 'time': time_val}
                            # Use .4f for float formatting in debug prints for consistency with output if values are non-NaN
                            score_debug_str = f"{score:.4f}" if not np.isnan(score) else "NaN"
                            time_debug_str = f"{time_val:.4f}" if not np.isnan(time_val) else "NaN"
                            print(f"DEBUG: Stored metrics for '{method_applied_csv}': score={score_debug_str}, time={time_debug_str}s")
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Could not parse score/time for '{method_applied_csv}' in '{base_instance_name}'. Row: {row}. Error: {e}")
                            # Store NaN to indicate data was found but unparseable
                            if method_applied_csv not in metrics: 
                                metrics[method_applied_csv] = {'score': float('nan'), 'time': float('nan')}
                    # else: # This can be noisy if many methods in CSV not in METHODS_TO_VISUALIZE
                    #     if method_applied_csv: # Only log if method_applied_csv is not empty
                    #         print(f"DEBUG: Method '{method_applied_csv}' (for instance '{base_instance_name}') is not in METHODS_TO_VISUALIZE. Skipping.")
            
    except FileNotFoundError: # Should be caught by os.path.exists, but good practice
        print(f"Error: Summary CSV file definitely not found at '{csv_path}' (Exception).") # Should not happen if exists check passed
    except Exception as e:
        print(f"Error reading or parsing CSV file '{csv_path}': {e}")
        # import traceback # Keep this commented unless deep debugging is needed by user
        # traceback.print_exc() # Provides full call stack for the exception

    print(f"DEBUG: Final loaded metrics dictionary for '{base_instance_name}': {metrics}")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Visualize solutions for multiple methods for a CVRP-LIB instance on subplots.")
    parser.add_argument('--base_instance_name', type=str, required=True,
                        help="Base name of the VRP instance (e.g., X-n101-k25).")
    parser.add_argument('--batch_run_dir', type=str, required=True,
                        help="Path to the batch run output directory (e.g., D:/Data-Proccess/cvrp_lib_integration/run_outputs_batch/batch_run_mm_20250517_185037).")
    parser.add_argument('--problem_type', type=str, default="CVRP",
                        help="The VRP problem type (default: CVRP).")
    parser.add_argument('--lang', type=str, default="en", choices=['en', 'zh'],
                        help="Language for visualization (en or zh, default: zh).")
    parser.add_argument('--show_annotations', action='store_true',
                        help="Show detailed annotations on the plot.")

    args = parser.parse_args()

    vrp_file_path = os.path.join(CVRPLIB_BASE_DIR, f"{args.base_instance_name}.vrp")
    if not os.path.exists(vrp_file_path):
        print(f"Error: VRP instance file not found at {vrp_file_path}")
        return

    # 1. Load and parse the VRP instance (once)
    print(f"Loading VRP instance from: {vrp_file_path}")
    parsed_dict = parse_vrp_file(vrp_file_path)
    if not parsed_dict:
        print(f"Failed to parse VRP file: {vrp_file_path}")
        return
    
    instance_tuple, problem_type_from_parser = to_internal_tuple(parsed_dict)
    if not instance_tuple:
        print(f"Failed to convert parsed VRP data to internal tuple format for: {vrp_file_path}")
        return
    
    effective_problem_type = args.problem_type if args.problem_type else problem_type_from_parser
    if not effective_problem_type:
        print("Error: Problem type could not be determined. Please specify with --problem_type.")
        return

    print(f"Successfully loaded and converted instance: {args.base_instance_name}. Effective problem type: {effective_problem_type}")
    # print_vrp_instance_info(instance_tuple, effective_problem_type, lang=args.lang) # Print info once

    # Load metrics from CSV
    method_metrics = load_method_metrics_from_csv(args.batch_run_dir, args.base_instance_name)

    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 18)) # Adjusted figsize for better layout
    axes = axes.flatten() # Flatten for easier iteration

    fig.suptitle(f"Solutions for {args.base_instance_name} ({effective_problem_type})", fontsize=16)

    for i, method_name in enumerate(METHODS_TO_VISUALIZE):
        ax = axes[i]
        # ax.set_title(method_name) # Set title for the subplot early

        # Prepare title string with metrics
        title_parts = [method_name]
        if method_name in method_metrics:
            metrics_for_method = method_metrics[method_name]
            score = metrics_for_method.get('cost', float('nan'))
            time = metrics_for_method.get('time_total_sec', float('nan'))
            score_str = f"{score:.2f}" if not np.isnan(score) else "N/A"
            time_str = f"{time:.2f}s" if not np.isnan(time) else "N/A"
            title_parts.append(f"Cost: {score_str}, Time: {time_str}")
        else:
            title_parts.append("Metrics N/A")
        ax.set_title("\n".join(title_parts), fontsize=10) # Use smaller fontsize for multi-line title

        solution_pkl_path = os.path.join(
            args.batch_run_dir,
            args.base_instance_name,
            method_name,
            f"{args.base_instance_name}_{method_name}_path.pkl"
        )

        print(f"\n--- Processing Method: {method_name} ---")
        if not os.path.exists(solution_pkl_path):
            print(f"Solution PKL file not found for {method_name} at: {solution_pkl_path}")
            ax.text(0.5, 0.5, f"Solution file not found\n{solution_pkl_path}", 
                    ha='center', va='center', fontsize=9, color='red', wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        print(f"Loading solution from: {solution_pkl_path}")
        try:
            with open(solution_pkl_path, 'rb') as f:
                loaded_solution_data = pickle.load(f)
        except Exception as e:
            print(f"Error loading solution from PKL file {solution_pkl_path}: {e}")
            ax.text(0.5, 0.5, f"Error loading solution\n{solution_pkl_path}\n{e}", 
                    ha='center', va='center', fontsize=9, color='red', wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        solution_for_viz = None
        if isinstance(loaded_solution_data, tuple) and len(loaded_solution_data) == 2:
            solution_for_viz = loaded_solution_data
            score_display = f"{loaded_solution_data[0]:.2f}" if isinstance(loaded_solution_data[0], float) and not np.isnan(loaded_solution_data[0]) else str(loaded_solution_data[0])
            print(f"Solution loaded as (score, path_list). Cost: {score_display}")
        elif isinstance(loaded_solution_data, list):
            print("Solution loaded as a path list. Using NaN for cost.")
            solution_for_viz = (float('nan'), loaded_solution_data)
        else:
            print(f"Error: Loaded solution data from {solution_pkl_path} is in an unrecognized format: {type(loaded_solution_data)}")
            ax.text(0.5, 0.5, f"Unrecognized solution format\n{solution_pkl_path}", 
                    ha='center', va='center', fontsize=9, color='red', wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        print(f"Visualizing solution for {method_name}...")
        try:
            # Calculate number of vehicles/routes from the path sequence
            num_vehicles = 0
            if solution_for_viz and len(solution_for_viz) > 1 and solution_for_viz[1]:
                path_sequence = solution_for_viz[1]
                # Count routes: a route ends when it returns to the depot (0)
                # or the sequence ends. Each segment between 0s (or start/end) is a route.
                # A simple way is to count non-empty segments after splitting by 0.
                current_segment = []
                vehicle_count_for_this_solution = 0
                for node_idx in path_sequence:
                    if node_idx == 0:
                        if current_segment: # End of a route segment
                            vehicle_count_for_this_solution += 1
                            current_segment = []
                    else:
                        current_segment.append(node_idx)
                if current_segment: # Account for the last route if it doesn't end with a depot visit in the list
                    vehicle_count_for_this_solution += 1
                num_vehicles = vehicle_count_for_this_solution
            
            # Construct the custom title here
            display_method_name = METHOD_DISPLAY_NAME_MAP.get(method_name, method_name) # Use mapped name or default to internal

            metrics_strings = []
            metrics_strings.append(f"Vehicles: {num_vehicles if num_vehicles > 0 else 'N/A'}")

            if method_name in method_metrics: # metrics are keyed by internal method name
                metrics_for_method = method_metrics[method_name]
                score = metrics_for_method.get('score', float('nan'))
                time = metrics_for_method.get('time', float('nan'))
                score_str = f"{score:.2f}" if not np.isnan(score) else "N/A"
                time_str = f"{time:.2f}s" if not np.isnan(time) else "N/A"
                metrics_strings.append(f"Cost: {score_str}")
                metrics_strings.append(f"Time: {time_str}")
            else:
                metrics_strings.append("Cost: N/A")
                metrics_strings.append("Time: N/A")
            
            custom_subplot_title = f"{display_method_name} - {', '.join(metrics_strings)}"
            
            # Pass the custom title to the visualize_solution function
            visualize_solution(
                instance_tuple,
                solution_for_viz,
                effective_problem_type,
                lang=args.lang,
                show_annotations=args.show_annotations,
                ax=ax,  # Pass the subplot axis
                custom_title=custom_subplot_title # Pass the fully constructed custom title
            )
        except Exception as e:
            print(f"Error visualizing solution for {method_name}: {e}")
            ax.text(0.5, 0.5, f"Error visualizing solution\n{method_name}\n{e}", 
                    ha='center', va='center', fontsize=9, color='red', wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()
    print("\nAll visualizations complete.")

if __name__ == "__main__":
    main() 