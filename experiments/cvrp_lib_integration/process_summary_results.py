import csv
import os
import math

# --- Configuration ---
INPUT_CSV_PATH = 'cvrp_lib_integration/run_outputs_batch/batch_run_mm_20250517_185037/summary_results_multimethod.csv'
# Output file will be in the same directory as the input, named 'summary_comparison.csv'

# Define method keys for easier access and consistency
M_DS = 'DirectSolver'
M_DO = 'DirectORTools'
M_PSA = 'PartitionSolver_adaptive'
M_POTA = 'PartitionORTools_adaptive'
M_OPT = 'OptimalSolution'

# Methods whose cost and time we want as direct columns
PRIMARY_METHODS = [M_DS, M_DO, M_PSA, M_POTA]
# Methods to be used as benchmarks for gap calculations
BENCHMARK_METHODS = [M_DS, M_DO, M_OPT]


def safe_float(value_str):
    """Converts a string to a float, handling 'inf' and potential errors."""
    if isinstance(value_str, (int, float)): # Already a number
        return float(value_str)
    if isinstance(value_str, str):
        value_str_lower = value_str.lower()
        if 'inf' in value_str_lower:
            return float('inf') if '-' not in value_str_lower else float('-inf')
        try:
            return float(value_str)
        except ValueError:
            return float('nan')
    return float('nan') # Default for other types or unhandled cases

def calculate_gap(cost_a, cost_b):
    """Calculates the percentage gap: (A - B) / B * 100."""
    cost_a_f = safe_float(cost_a)
    cost_b_f = safe_float(cost_b)

    if math.isnan(cost_a_f) or math.isnan(cost_b_f) or math.isinf(cost_a_f):
        return float('nan')
    
    if cost_b_f == 0: # Avoid division by zero
        if cost_a_f == 0: return 0.0 # Both are zero
        return float('inf') if cost_a_f > 0 else float('-inf') if cost_a_f < 0 else float('nan') # Should ideally not happen if B is cost

    if math.isinf(cost_b_f):
        # If benchmark is infinite
        if math.isinf(cost_a_f) and (cost_a_f > 0) == (cost_b_f > 0): # Both inf with same sign
            return 0.0 
        # If A is finite and B is infinite, A is "infinitely better" or "infinitely worse"
        # For cost, if B is +inf, and A is finite, A is better. Gap could be -100%
        # But to keep it simple, if benchmark is inf and method is not also inf same sign, consider N/A
        return float('nan') 

    return (cost_b_f - cost_a_f) / cost_b_f * 100

def main():
    print(f"Processing CSV file: {INPUT_CSV_PATH}")
    
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input CSV file not found at {INPUT_CSV_PATH}")
        return

    base_dir = os.path.dirname(INPUT_CSV_PATH)
    output_csv_path = os.path.join(base_dir, 'summary_comparison.csv')

    all_rows = []
    with open(INPUT_CSV_PATH, 'r', newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            all_rows.append(row)

    if not all_rows:
        print("Input CSV is empty.")
        return

    # Group data by problem_name
    instance_data_map = {}
    for row in all_rows:
        problem_name = row.get('problem_name')
        if problem_name:
            if problem_name not in instance_data_map:
                instance_data_map[problem_name] = []
            instance_data_map[problem_name].append(row)

    processed_results = []
    output_headers = ['name', 'problem_size_N']

    # Add cost and time headers for primary methods
    for method_key in PRIMARY_METHODS:
        output_headers.append(f'cost_{method_key}')
    output_headers.append(f'cost_{M_OPT}') # Optimal cost
    for method_key in PRIMARY_METHODS:
        output_headers.append(f'time_{method_key}')
    
    # Add gap headers
    for method_to_eval in PRIMARY_METHODS:
        for benchmark_method in BENCHMARK_METHODS:
            if method_to_eval == benchmark_method: continue # No gap against itself needed here
            output_headers.append(f'gap_{method_to_eval}_vs_{benchmark_method}')

    for problem_name, rows_for_instance in instance_data_map.items():
        output_row = {'name': problem_name}
        
        # Get problem_size_N (should be consistent for the instance)
        try:
            output_row['problem_size_N'] = int(rows_for_instance[0].get('problem_size_N', 0))
        except ValueError:
            output_row['problem_size_N'] = 0 # Or handle error appropriately

        # Store metrics for each method found for this instance
        metrics_by_method = {}
        for row in rows_for_instance:
            method_applied = row.get('method_applied')
            if method_applied:
                metrics_by_method[method_applied] = {
                    'cost': safe_float(row.get('cost')),
                    'time': safe_float(row.get('time_total_sec'))
                }
        
        # Populate costs and times for primary methods and optimal
        all_method_costs = {} # To store costs for gap calculation

        for method_key in PRIMARY_METHODS:
            cost = metrics_by_method.get(method_key, {}).get('cost', float('nan'))
            time = metrics_by_method.get(method_key, {}).get('time', float('nan'))
            output_row[f'cost_{method_key}'] = cost
            output_row[f'time_{method_key}'] = time
            all_method_costs[method_key] = cost
            
        cost_optimal = metrics_by_method.get(M_OPT, {}).get('cost', float('nan'))
        output_row[f'cost_{M_OPT}'] = cost_optimal
        all_method_costs[M_OPT] = cost_optimal

        # Calculate and populate gaps
        for method_to_eval in PRIMARY_METHODS:
            cost_eval = all_method_costs.get(method_to_eval, float('nan'))
            for benchmark_method in BENCHMARK_METHODS:
                if method_to_eval == benchmark_method: continue
                
                cost_benchmark = all_method_costs.get(benchmark_method, float('nan'))
                gap = calculate_gap(cost_eval, cost_benchmark)
                output_row[f'gap_{method_to_eval}_vs_{benchmark_method}'] = gap
                
        processed_results.append(output_row)

    # Sort by problem_size_N
    processed_results.sort(key=lambda x: (x.get('problem_size_N', 0), x.get('name', '')))

    # Write to output CSV
    with open(output_csv_path, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_headers, extrasaction='ignore')
        writer.writeheader()
        for row_data in processed_results:
            # Format floats for writing
            formatted_row = {}
            for key, val in row_data.items():
                if isinstance(val, float):
                    if math.isnan(val): formatted_row[key] = 'NaN'
                    elif math.isinf(val): formatted_row[key] = 'inf' if val > 0 else '-inf'
                    else: formatted_row[key] = f"{val:.2f}" # Format to 2 decimal places
                else:
                    formatted_row[key] = val
            writer.writerow(formatted_row)
            
    print(f"Successfully processed data and saved to: {output_csv_path}")

if __name__ == '__main__':
    main() 