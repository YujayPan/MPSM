import csv
import os
import math
from collections import defaultdict

# --- Configuration ---
INPUT_COMPARISON_CSV_PATH = 'cvrp_lib_integration/run_outputs_batch/batch_run_mm_20250517_185037/summary_comparison.csv'
TOP_N = 30

# MODIFIED: Only analyze these two gap columns
GAP_COLUMNS_TO_ANALYZE = [
    'gap_PartitionSolver_adaptive_vs_DirectSolver',
    'gap_PartitionSolver_adaptive_vs_DirectORTools'
    # 'gap_PartitionSolver_adaptive_vs_OptimalSolution' # REMOVED
]

def safe_float_from_csv(value_str):
    """Converts a string from CSV (possibly 'NaN', 'inf') to a float."""
    if isinstance(value_str, (int, float)):
        return float(value_str)
    if isinstance(value_str, str):
        val_low = value_str.lower()
        if val_low == 'nan':
            return float('nan')
        if val_low == 'inf':
            return float('inf')
        if val_low == '-inf':
            return float('-inf')
        try:
            return float(value_str)
        except ValueError:
            return float('nan')
    return float('nan')

def main():
    print(f"Analyzing comparison CSV file: {INPUT_COMPARISON_CSV_PATH}")

    if not os.path.exists(INPUT_COMPARISON_CSV_PATH):
        print(f"Error: Input CSV file not found at {INPUT_COMPARISON_CSV_PATH}")
        return

    all_instance_data = []
    try:
        with open(INPUT_COMPARISON_CSV_PATH, 'r', newline='') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                instance_entry = {'name': row.get('name')}
                # Ensure we also read the OptimalSolution gap if needed for other reports, 
                # even if not directly analyzed for top N here.
                # For this specific request, we only need the two specified.
                for gap_col in GAP_COLUMNS_TO_ANALYZE: # Only populate the two we need for analysis
                    instance_entry[gap_col] = safe_float_from_csv(row.get(gap_col))
                all_instance_data.append(instance_entry)
    except Exception as e:
        print(f"Error reading or parsing CSV: {e}")
        return

    if not all_instance_data:
        print("No data found in the comparison CSV.")
        return

    top_instances_by_gap = defaultdict(list)
    all_top_instance_sets = {}

    for gap_col_name in GAP_COLUMNS_TO_ANALYZE:
        print(f"\n--- Top {TOP_N} instances for: {gap_col_name} ---")
        
        valid_data_for_gap = [d for d in all_instance_data if d.get(gap_col_name) is not None and not math.isnan(d[gap_col_name]) and not math.isinf(d[gap_col_name])]
        
        if not valid_data_for_gap:
            print(f"  No valid (non-NaN, non-inf) gap data found for {gap_col_name}.")
            all_top_instance_sets[gap_col_name] = set()
            continue
            
        sorted_by_gap = sorted(valid_data_for_gap, key=lambda x: x[gap_col_name], reverse=True)
        
        current_top_n = sorted_by_gap[:TOP_N]
        top_instances_by_gap[gap_col_name] = current_top_n
        all_top_instance_sets[gap_col_name] = set(instance['name'] for instance in current_top_n)

        # MODIFIED: Print only the two analyzed gap columns
        print(f"  Rank | Instance Name         | {GAP_COLUMNS_TO_ANALYZE[0]:<45} | {GAP_COLUMNS_TO_ANALYZE[1]:<45}")
        print("  " + "-" * 110) # Adjusted line length
        for i, instance in enumerate(current_top_n):
            gap_val_1 = instance.get(GAP_COLUMNS_TO_ANALYZE[0], float('nan'))
            gap_val_2 = instance.get(GAP_COLUMNS_TO_ANALYZE[1], float('nan'))
            print(f"  {i+1:<4} | {instance['name']:<20} | {gap_val_1:<45.2f} | {gap_val_2:<45.2f}")

    # Find overlapping instances between the TWO specified gap lists
    print("\n--- Overlapping Instances in Top Lists ---")
    # Ensure we have entries for both keys before trying to intersect
    set1_name = GAP_COLUMNS_TO_ANALYZE[0]
    set2_name = GAP_COLUMNS_TO_ANALYZE[1]

    if set1_name in all_top_instance_sets and set2_name in all_top_instance_sets:
        set1 = all_top_instance_sets[set1_name]
        set2 = all_top_instance_sets[set2_name]
        
        if not set1 or not set2:
            print(f"  One or both top lists for '{set1_name}' or '{set2_name}' are empty. Cannot find overlaps.")
        else:
            common_instances = set1.intersection(set2)
            if common_instances:
                print(f"  Instances present in the Top {TOP_N} of BOTH '{set1_name}' AND '{set2_name}':")
                for name in sorted(list(common_instances)):
                    print(f"    - {name}")
            else:
                print(f"  No instances found that are in the Top {TOP_N} of both '{set1_name}' and '{set2_name}' simultaneously.")
    else:
        print(f"  Could not find top instance sets for both required gap columns: '{set1_name}' and '{set2_name}'. Overlap analysis skipped.")
    
    print("\nAnalysis complete.")

if __name__ == '__main__':
    main() 