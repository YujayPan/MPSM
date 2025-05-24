import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# Chinese instructions:
# This script supports displaying data for a specific problem_type by selecting it through command line arguments.
# It also combines the score and gap curves for each method into a single plot (two subplots), with consistent colors for each method.
# Added --y_map parameter, which can be linear/sqrt/square/log, to map the y-axis data for better visualization.
# Added --x_map parameter, which can be linear/log/both, to decide the x-axis scaling and subplot layout.
# Added --gap_calculation_method parameter, which can be mixed_score/average_of_types, to decide the gap calculation method for "All Variants".

def apply_y_map(y, y_map):
    """
    Apply y-axis mapping function to the data.
    y_map: 'linear', 'sqrt', 'square', 'log'
    """
    if y_map == 'linear':
        return y
    elif y_map == 'sqrt':
        # sqrt for both positive and negative values
        return np.sign(y) * np.sqrt(np.abs(y))
    elif y_map == 'square':
        return np.square(y)
    elif y_map == 'log':
        # log1p for both positive and negative values, keep sign
        return np.sign(y) * np.log1p(np.abs(y))
    else:
        return y

def visualize_combined_scores(csv_filepath, problem_type=None, y_map='linear', x_map='linear', gap_calculation_method='average_of_types', problem_sizes_filter=None, selected_methods_only=False):
    """
    Visualizes scores and times, and prints a summary table with mean, std, and gaps.
    Reads aggregated data (mean, std) from the CSV file.
    """
    try:
        df_all_types_aggregated = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Aggregated CSV file not found at '{csv_filepath}'")
        print("Please run aggregate_data.py first to generate this file.")
        return
    except Exception as e:
        print(f"Error reading aggregated CSV file: {e}")
        return

    # Rename columns for consistency if they come from aggregate_data.py
    # Expected: problem_type, problem_size, method, score_mean, score_std, total_time_seconds_mean, total_time_seconds_std
    # For plotting, we'll primarily use the _mean columns as the base for 'score' and 'total_time_seconds'

    unique_problem_types_in_file = df_all_types_aggregated['problem_type'].unique()
    
    def get_display_method_name(original_name):
        if original_name == "DirectSolver":
            return "Solver"
        if original_name == "DirectORTools": # Note: ORTools casing from user's edit
            return "ORTools"
        
        # Handling Partition-based methods
        # Expected format: PartitionSolver_METHOD or PartitionORTools_METHOD
        if original_name.startswith("PartitionSolver_"):
            parts = original_name.split("_")
            if len(parts) == 2: # e.g., PartitionSolver_m1
                merge_method = parts[1]
                return f"MPSM_{merge_method} + Solver"
        elif original_name.startswith("PartitionORTools_"):
            parts = original_name.split("_")
            if len(parts) == 2: # e.g., PartitionORTools_adaptive
                merge_method = parts[1]
                return f"MPSM_{merge_method} + ORTools"
        
        return original_name # Fallback if no specific mapping found

    title_variant_specifier = "(All Variants)"
    current_df_aggregated = df_all_types_aggregated.copy()
    if problem_type is not None:
        current_df_aggregated = df_all_types_aggregated[df_all_types_aggregated['problem_type'].isin(problem_type)].copy()
        if current_df_aggregated.empty:
            print(f"No data for problem_type(s) '{', '.join(problem_type)}' after filtering aggregated data.")
            return
        title_variant_specifier = f"({', '.join(problem_type)})"
    
    if problem_sizes_filter:
        # Ensure problem_sizes_filter contains integers for comparison with 'problem_size' column
        problem_sizes_filter_int = [int(ps) for ps in problem_sizes_filter]
        current_df_aggregated = current_df_aggregated[current_df_aggregated['problem_size'].isin(problem_sizes_filter_int)]
        if current_df_aggregated.empty:
            print(f"No data for problem_size(s) '{', '.join(map(str, problem_sizes_filter_int))}' after filtering.")
            return
        # Append to title_variant_specifier if it's not the default "All Variants"
        if title_variant_specifier == "(All Variants)":
            title_variant_specifier = f"(Sizes: {', '.join(map(str, problem_sizes_filter_int))})"
        else: # Already has problem types
            title_variant_specifier = title_variant_specifier[:-1] + f"; Sizes: {', '.join(map(str, problem_sizes_filter_int))})"

    if selected_methods_only:
        selected_methods_list = ["DirectSolver", "DirectORTools", "PartitionSolver_adaptive", "PartitionORTools_adaptive"]
        current_df_aggregated = current_df_aggregated[current_df_aggregated['method'].isin(selected_methods_list)]
        if current_df_aggregated.empty:
            print(f"No data for the selected methods (DirectSolver, DirectORTools, PartitionSolver_adaptive, PartitionORTools_adaptive) after filtering.")
            return
        # Append to title_variant_specifier
        method_filter_note = "Core Methods"
        if title_variant_specifier == "(All Variants)":
            title_variant_specifier = f"({method_filter_note})"
        elif "Sizes:" in title_variant_specifier and "Types:" not in title_variant_specifier and problem_type is None : # Only size filter was applied
             title_variant_specifier = title_variant_specifier[:-1] + f"; {method_filter_note})"
        else: # Already has problem types or problem types and sizes
            title_variant_specifier = title_variant_specifier[:-1] + f"; {method_filter_note})"

    if current_df_aggregated.empty:
        print("No data available for plotting/tabulation after filtering.")
        return

    # For PLOTTING: use mean values for score and time
    try:
        # Create 'score' and 'total_time_seconds' from mean columns for plotting functions
        plot_df = current_df_aggregated.rename(columns={
            'score_mean': 'score', 
            'total_time_seconds_mean': 'total_time_seconds'
        })
        grouped_scores_for_plot = plot_df.groupby(['problem_size', 'method'])['score'].mean().unstack() # Effectively selects score_mean
        grouped_times_for_plot = plot_df.groupby(['problem_size', 'method'])['total_time_seconds'].mean().unstack() # Effectively selects time_mean
    except Exception as e:
        print(f"Error processing data for plots: {e}")
        return
    
    if grouped_scores_for_plot.empty or grouped_times_for_plot.empty:
        print(f"No data to plot for scores or times after grouping from aggregated data.")
        # Don't return yet, table might still be possible if one is empty but not the other

    method_list = current_df_aggregated['method'].unique().tolist()
    if not method_list:
        print("No methods found in the aggregated data.")
        return
        
    color_map = plt.get_cmap('tab10')
    method_colors = {method: color_map(i % 10) for i, method in enumerate(method_list)}
    marker_shapes = ['o', 's', '^', 'D', 'v', 'p', '*', '+', 'x', 'h', 'H']
    method_markers = {method: marker_shapes[i % len(marker_shapes)] for i, method in enumerate(method_list)}

    # --- Plotting Gap Data Preparation (based on MEAN values) ---
    # Cost Gap Data (relative to DirectSolver's score_mean)
    final_gap_data = pd.DataFrame() # Cost Gap
    if not grouped_scores_for_plot.empty and 'DirectSolver' in grouped_scores_for_plot.columns:
        benchmark_scores_mean = grouped_scores_for_plot['DirectSolver']
        temp_gap_data = pd.DataFrame(index=grouped_scores_for_plot.index)
        for m_iter in grouped_scores_for_plot.columns:
            if m_iter == 'DirectSolver': continue
            m_scores = grouped_scores_for_plot[m_iter]
            aligned_bench, aligned_m = benchmark_scores_mean.align(m_scores, join='inner')
            if not aligned_bench.empty: temp_gap_data[m_iter] = (aligned_bench - aligned_m) / aligned_bench * 100
        temp_gap_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        final_gap_data = temp_gap_data
    elif not grouped_scores_for_plot.empty:
        print("'DirectSolver' not found for Cost Gap calculation in plots.")

    # Time Gap Data (relative to DirectORTools' total_time_seconds_mean)
    time_gap_data = pd.DataFrame() # Time Gap for plots
    if not grouped_times_for_plot.empty and 'DirectORTools' in grouped_times_for_plot.columns:
        benchmark_times_mean = grouped_times_for_plot['DirectORTools']
        temp_time_gap_data = pd.DataFrame(index=grouped_times_for_plot.index)
        for m_iter in grouped_times_for_plot.columns:
            if m_iter == 'DirectORTools': continue
            m_times = grouped_times_for_plot[m_iter]
            aligned_bench, aligned_m_time = benchmark_times_mean.align(m_times, join='inner')
            if not aligned_bench.empty: temp_time_gap_data[m_iter] = (aligned_bench - aligned_m_time) / aligned_bench * 100
        temp_time_gap_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        time_gap_data = temp_time_gap_data
    elif not grouped_times_for_plot.empty:
        print("'DirectORTools' not found for Time Gap calculation in plots.")

    # Apply y_map (rest of plotting logic remains largely the same using these _mean based DFs)
    mapped_grouped_scores = grouped_scores_for_plot.copy() if not grouped_scores_for_plot.empty else pd.DataFrame()
    if not mapped_grouped_scores.empty:
        for m_iter in method_list: 
            if m_iter in mapped_grouped_scores.columns: mapped_grouped_scores[m_iter] = apply_y_map(mapped_grouped_scores[m_iter], y_map)
    
    mapped_final_gap_data = final_gap_data.copy() if not final_gap_data.empty else pd.DataFrame()
    if not mapped_final_gap_data.empty:
        for m_iter in mapped_final_gap_data.columns: mapped_final_gap_data[m_iter] = apply_y_map(final_gap_data[m_iter], y_map)
    
    mapped_grouped_times = grouped_times_for_plot.copy() if not grouped_times_for_plot.empty else pd.DataFrame()
    if not mapped_grouped_times.empty:
        for m_iter in method_list:
            if m_iter in mapped_grouped_times.columns: mapped_grouped_times[m_iter] = apply_y_map(mapped_grouped_times[m_iter], y_map)

    mapped_time_gap_data = time_gap_data.copy() if not time_gap_data.empty else pd.DataFrame()
    if not mapped_time_gap_data.empty:
        for m_iter in mapped_time_gap_data.columns: mapped_time_gap_data[m_iter] = apply_y_map(time_gap_data[m_iter], y_map)

    unique_sizes_scores = sorted(grouped_scores_for_plot.index.unique()) if not grouped_scores_for_plot.empty else []
    unique_sizes_gaps = sorted(final_gap_data.index.unique()) if not final_gap_data.empty else []
    unique_sizes_times = sorted(grouped_times_for_plot.index.unique()) if not grouped_times_for_plot.empty else []
    unique_sizes_time_gap = sorted(time_gap_data.index.unique()) if not time_gap_data.empty else []
    unique_sizes = sorted(list(set(unique_sizes_scores) | set(unique_sizes_gaps) | set(unique_sizes_times) | set(unique_sizes_time_gap)))

    # --- END OF PLOTTING DATA PREPARATION ---

    # --- IMAGE-STYLE TABLE PREPARATION ---
    table_data_rows = []
    # Use current_df_aggregated which has all mean/std columns
    # Ensure problem_size is int for consistent indexing later in pivot_table creation
    current_df_aggregated['problem_size'] = current_df_aggregated['problem_size'].astype(int)

    # Calculate Gaps for table using mean values from current_df_aggregated
    # Cost Gap vs DirectSolver
    ds_scores = current_df_aggregated[current_df_aggregated['method'] == 'DirectSolver']
    ds_scores = ds_scores.set_index('problem_size')['score_mean'].to_dict() if not ds_scores.empty else {}
    # Time Gap vs DirectORTools
    dot_times = current_df_aggregated[current_df_aggregated['method'] == 'DirectORTools']
    dot_times = dot_times.set_index('problem_size')['total_time_seconds_mean'].to_dict() if not dot_times.empty else {}

    for index, row in current_df_aggregated.iterrows():
        size = row['problem_size']
        method_orig = row['method']
        display_name = get_display_method_name(method_orig)

        s_mean = row['score_mean']
        s_std = row['score_std']
        t_mean = row['total_time_seconds_mean']
        t_std = row['total_time_seconds_std']

        len_str = f"{s_mean:.2f}" if pd.isna(s_std) or s_std == 0 else f"{s_mean:.2f} ± {s_std:.2f}"
        time_str = f"{t_mean:.2f}" if pd.isna(t_std) or t_std == 0 else f"{t_mean:.2f} ± {t_std:.2f}"

        table_data_rows.append({'Problem Size': size, 'METRIC': 'LENGTH', 'Method_Display': display_name, 'Value': len_str})
        table_data_rows.append({'Problem Size': size, 'METRIC': 'TIME', 'Method_Display': display_name, 'Value': time_str})

        # Cost Gap
        s_gap_val = '-'
        if method_orig == 'DirectSolver': s_gap_val = "0.00%"
        elif ds_scores.get(size) and pd.notna(s_mean) and ds_scores[size] != 0:
            s_gap_calc = (ds_scores[size] - s_mean) / ds_scores[size] * 100
            s_gap_val = f"{s_gap_calc:.2f}%"
        table_data_rows.append({'Problem Size': size, 'METRIC': 'L_GAP (%)', 'Method_Display': display_name, 'Value': s_gap_val})

        # Time Gap
        t_gap_val = '-'
        if method_orig == 'DirectORTools': t_gap_val = "0.00%"
        elif dot_times.get(size) and pd.notna(t_mean) and dot_times[size] != 0:
            t_gap_calc = (dot_times[size] - t_mean) / dot_times[size] * 100
            t_gap_val = f"{t_gap_calc:.2f}%"
        table_data_rows.append({'Problem Size': size, 'METRIC': 'T_GAP (%)', 'Method_Display': display_name, 'Value': t_gap_val})

    final_summary_table_df = pd.DataFrame(table_data_rows)
    pivoted_summary_table = pd.DataFrame()
    if not final_summary_table_df.empty:
        try:
            pivoted_summary_table = final_summary_table_df.pivot_table(
                index=['Problem Size', 'METRIC'], 
                columns='Method_Display', 
                values='Value', 
                aggfunc='first' # To handle potential duplicates if any, though logic aims to avoid them
            )
            metric_order = ['LENGTH', 'L_GAP (%)', 'TIME', 'T_GAP (%)']
            pivoted_summary_table = pivoted_summary_table.reindex(metric_order, level='METRIC')
            
            ordered_display_methods = [get_display_method_name(m) for m in method_list] # Use unique methods from current_df_aggregated
            existing_display_methods_in_order = [m for m in ordered_display_methods if m in pivoted_summary_table.columns]
            if existing_display_methods_in_order:
                 pivoted_summary_table = pivoted_summary_table[existing_display_methods_in_order]

        except Exception as e:
            print(f"Error creating final pivoted summary table: {e}")
    # --- END OF IMAGE-STYLE TABLE PREPARATION ---

    # --- PLOTTING LOGIC (Remains same, uses *_for_plot and *_gap_data DFs) ---
    if x_map == 'both':
        x_scales_to_plot = ['log', 'linear']
    else:
        x_scales_to_plot = [x_map] 

    for current_x_scale in x_scales_to_plot:
        fig, axes = plt.subplots(4, 1, figsize=(12, 20), sharey=False) 
        if len(x_scales_to_plot) > 1: fig.suptitle(f"X-axis scale: {current_x_scale.upper()}", fontsize=14)

        # Plotting calls for axes[0], axes[1], axes[2], axes[3] 
        # Cost plot (top)
        ax = axes[0]
        if not mapped_grouped_scores.empty:
            for method_orig_name in method_list: 
                if method_orig_name in mapped_grouped_scores.columns:
                    display_name = get_display_method_name(method_orig_name)
                    ax.plot(mapped_grouped_scores.index, mapped_grouped_scores[method_orig_name], 
                            marker=method_markers[method_orig_name], label=display_name, 
                            color=method_colors[method_orig_name], alpha=0.7, linestyle='-')
        if current_x_scale == 'log': ax.set_xscale('log')
        ax.set_title(f"Cost (Total Route Length)")
        ax.set_ylabel(f'Average Cost (y_map: {y_map})')
        ax.legend(title='Method'); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xticks(unique_sizes); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        # Cost Gap plot (second)
        ax = axes[1]
        if not mapped_final_gap_data.empty:
            for method_orig_name in mapped_final_gap_data.columns:
                if method_orig_name in method_colors and method_orig_name in method_markers:
                    display_name = get_display_method_name(method_orig_name)
                    ax.plot(mapped_final_gap_data.index, mapped_final_gap_data[method_orig_name], 
                            marker=method_markers[method_orig_name], label=display_name, 
                            color=method_colors[method_orig_name], alpha=0.7, linestyle='-')
        if current_x_scale == 'log': ax.set_xscale('log')
        ax.set_title(f"Cost Gap (Improvement compared to DirectSolver)")
        ax.set_ylabel(f'Improvement Gap (%) (y_map: {y_map})'); ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax.legend(title='Method'); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xticks(unique_sizes); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        if mapped_final_gap_data.empty: ax.text(0.5, 0.5, 'No score gap data available', ha='center', va='center', transform=ax.transAxes)

        # Time plot (third)
        ax = axes[2]
        if not mapped_grouped_times.empty:
            for method_orig_name in method_list:
                if method_orig_name in mapped_grouped_times.columns:
                    display_name = get_display_method_name(method_orig_name)
                    ax.plot(mapped_grouped_times.index, mapped_grouped_times[method_orig_name],
                            marker=method_markers[method_orig_name], label=display_name,
                            color=method_colors[method_orig_name], alpha=0.7, linestyle='-')
        if current_x_scale == 'log': ax.set_xscale('log')
        ax.set_title(f"Time")
        ax.set_ylabel(f'Avg Total Time (s) (y_map: {y_map})')
        ax.legend(title='Method'); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xticks(unique_sizes); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        if mapped_grouped_times.empty: ax.text(0.5, 0.5, 'No time data available', ha='center', va='center', transform=ax.transAxes)

        # Time Gap plot (bottom)
        ax = axes[3]
        if not mapped_time_gap_data.empty:
            for method_orig_name in mapped_time_gap_data.columns:
                if method_orig_name in method_colors and method_orig_name in method_markers:
                    display_name = get_display_method_name(method_orig_name)
                    ax.plot(mapped_time_gap_data.index, mapped_time_gap_data[method_orig_name],
                            marker=method_markers[method_orig_name], label=display_name,
                            color=method_colors[method_orig_name], alpha=0.7, linestyle='-')
        if current_x_scale == 'log': ax.set_xscale('log')
        ax.set_title(f"Time Gap (Improvement compared to DirectORTools)")
        ax.set_xlabel('Problem Size')
        ax.set_ylabel(f'Time Gap (%) (y_map: {y_map})'); ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax.legend(title='Method'); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xticks(unique_sizes); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        if mapped_time_gap_data.empty: ax.text(0.5, 0.5, 'No time gap data (DirectORTools missing or other issue)', ha='center', va='center', transform=ax.transAxes)
        
        if len(x_scales_to_plot) > 1: fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        else: fig.tight_layout()
        plt.show()
    # --- END OF PLOTTING LOGIC ---

    # --- PRINT FINAL SUMMARY TABLE ---
    if not pivoted_summary_table.empty:
        print("\n\n" + "="*30 + f" Results Summary: {title_variant_specifier} " + "="*30)
        # For multi-index, to_string() handles it well. Ensure METRIC order.
        print(pivoted_summary_table.to_string(na_rep="-"))
        print("\n" + "="*75)

        # --- SAVE FINAL SUMMARY TABLE TO CSV ---
        table_problem_type_str = "_".join(problem_type) if problem_type else "All_Variants"
        summary_table_filename = f"results/compare/summary_table_{table_problem_type_str}_y-{y_map}_x-{x_map}.csv"
        try:
            pivoted_summary_table.to_csv(summary_table_filename)
            print(f"Summary table saved to: {summary_table_filename}")
        except Exception as e:
            print(f"Error saving summary table to CSV: {e}")
        # --- END OF SAVE FINAL SUMMARY TABLE ---

    else:
        print("\nNo data available for the summary table.")
    # --- END OF PRINT FINAL SUMMARY TABLE ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize VRP method scores, gaps, and times from aggregated data.')
    parser.add_argument('--csv_file', type=str, default='results/compare/aggregated_results.csv', # Updated default
                        help='Path to the AGGREGATED results CSV file (default: results/compare/aggregated_results.csv)')
    parser.add_argument('--problem_type', type=str, nargs='+', default=None, 
                        help='Specify one or more problem_types to filter (e.g., CVRP OVRP). If not set, show all variants.')
    parser.add_argument('--y_map', type=str, default='linear', choices=['linear', 'sqrt', 'square', 'log'],
                        help='Y-axis mapping: linear, sqrt, square, log. Default: linear.')
    parser.add_argument('--x_map', type=str, default='linear', choices=['linear', 'log', 'both'],
                        help='X-axis mapping: linear, log, both. Default: linear.')
    parser.add_argument('--gap_calculation_method', type=str, default='average_of_types', choices=['mixed_score', 'average_of_types'],
                        help="Method for Gap calculation when 'All Variants' is selected. "
                             "'mixed_score': Gap based on scores averaged across all types. "
                             "'average_of_types': Gap is the average of Gaps calculated for each type individually (default for 'All Variants').")
    parser.add_argument('--problem_size', type=int, nargs='+', default=None,
                        help='Specify one or more problem_sizes to filter (e.g., 50 100 200). If not set, show all sizes.')
    parser.add_argument('--selected_methods_only', action='store_true',
                        help='If set, only plot/tabulate DirectSolver, DirectORTools, PartitionSolver_adaptive, PartitionORTools_adaptive.')

    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"IMPORTANT: The AGGREGATED CSV file path '{args.csv_file}' does not exist.")
        print("Please run aggregate_data.py first or specify the correct path using --csv_file.")
    else:
        visualize_combined_scores(args.csv_file, args.problem_type, args.y_map, args.x_map, args.gap_calculation_method, args.problem_size, args.selected_methods_only)
