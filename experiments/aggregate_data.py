import pandas as pd
import numpy as np
import argparse
import os

def aggregate_data(input_csv_path, output_csv_path):
    """
    Reads the results CSV, calculates mean and std for score and time
    for each (problem_type, problem_size, method) group, and saves
    the aggregated data to a new CSV file.
    """
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at '{input_csv_path}'")
        return
    except Exception as e:
        print(f"Error reading input CSV file: {e}")
        return

    # Ensure necessary columns are numeric
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['total_time_seconds'] = pd.to_numeric(df['total_time_seconds'], errors='coerce')
    df['problem_size'] = pd.to_numeric(df['problem_size'], errors='coerce')

    # Drop rows where key numeric data might be NaN after coercion, affecting aggregation
    df.dropna(subset=['problem_type', 'problem_size', 'method', 'score', 'total_time_seconds'], inplace=True)

    if df.empty:
        print("No valid data found in the input CSV after cleaning. Output file will not be created.")
        return

    print(f"Aggregating data from '{input_csv_path}'...")

    # Define aggregation functions
    agg_functions = {
        'score': ['mean', 'std'],
        'total_time_seconds': ['mean', 'std']
    }

    # Group and aggregate
    try:
        # Ensure problem_size is treated as a number for grouping if it isn't already reliable
        # This was already done above, but good to be mindful of its type before grouping
        aggregated_df = df.groupby(['problem_type', 'problem_size', 'method'], as_index=False).agg(agg_functions)
    except Exception as e:
        print(f"Error during data aggregation: {e}")
        return

    # Flatten the multi-level column index
    # e.g., ('score', 'mean') becomes 'score_mean'
    new_cols = []
    for col_tuple in aggregated_df.columns.values:
        if isinstance(col_tuple, tuple):
            # For aggregated columns like ('score', 'mean')
            if col_tuple[1]: # If the second part of tuple is not empty (i.e., mean, std)
                new_cols.append(f'{col_tuple[0]}_{col_tuple[1]}')
            else: # If it's a grouping key that was part of columns before as_index=False
                new_cols.append(col_tuple[0])
        else:
            # For grouping keys that are already single level (problem_type, problem_size, method)
            new_cols.append(col_tuple)
    aggregated_df.columns = new_cols

    try:
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        aggregated_df.to_csv(output_csv_path, index=False, float_format='%.4f')
        print(f"Aggregated data successfully saved to '{output_csv_path}'")
    except Exception as e:
        print(f"Error writing output CSV file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aggregate VRP results data to include mean and standard deviation.")
    parser.add_argument(
        '--input_csv', 
        type=str, 
        default='results/compare/results.csv', 
        help='Path to the input results CSV file (default: results/compare/results.csv)'
    )
    parser.add_argument(
        '--output_csv', 
        type=str, 
        default='results/compare/aggregated_results.csv', 
        help='Path to save the aggregated results CSV file (default: results/compare/aggregated_results.csv)'
    )
    args = parser.parse_args()

    aggregate_data(args.input_csv, args.output_csv) 