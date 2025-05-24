import pandas as pd
import argparse
import os
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze the adaptive target size results")
    parser.add_argument('--input_csv', type=str, required=True,
                        help="Path to the result CSV file")
    parser.add_argument('--output_dir', type=str, default='adaptive_target_analysis',
                        help="Directory to save the analysis results")
    return parser.parse_args()

def analyze_results(df: pd.DataFrame) -> Dict[Tuple[str, int], dict]:
    """Analyze the results and find the best target_node_count for each problem type and size.

    Args:
        df: DataFrame containing the results

    Returns:
        Dict[Tuple[str, int], dict]: Keys are (problem_type, size), values are the analysis results for that combination
    """
    results = {}
    
    # Group by problem type and size
    for (prob_type, size), group in df.groupby(['problem_type', 'problem_size']):
        # Calculate average score by target_node_count
        target_scores = group.groupby('target_node_count').agg({
            'score': ['mean', 'std', 'count'],  # Calculate average score, standard deviation, and sample count
            'num_subproblems': 'mean',  # Average number of subproblems
            'total_time_seconds': 'mean'  # Average total time
        })
        
        # Rename columns for easier access
        target_scores.columns = ['score_mean', 'score_std', 'sample_count', 
                               'avg_subproblems', 'avg_time']
        
        # Find the best target (lowest average score)
        best_target = target_scores['score_mean'].idxmin()
        best_score = target_scores.loc[best_target, 'score_mean']
        
        results[(prob_type, size)] = {
            'best_target': best_target,
            'best_score': best_score,
            'all_targets_data': target_scores,
            'num_instances': len(group['instance_index'].unique()),
            'avg_subproblems': target_scores.loc[best_target, 'avg_subproblems'],
            'avg_time': target_scores.loc[best_target, 'avg_time']
        }
    
    return results

def plot_target_comparison(results: Dict[Tuple[str, int], dict], output_dir: str):
    """Create a target size comparison plot for each problem type."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by problem type
    problem_types = sorted(set(k[0] for k in results.keys()))
    sizes = sorted(set(k[1] for k in results.keys()))
    
    for prob_type in problem_types:
        plt.figure(figsize=(12, 6))
        
        for size in sizes:
            if (prob_type, size) not in results:
                continue
                
            data = results[(prob_type, size)]['all_targets_data']
            plt.plot(data.index, data['score_mean'], 
                    marker='o', label=f'N={size}')
            
            # Add error bars
            plt.fill_between(data.index,
                           data['score_mean'] - data['score_std'],
                           data['score_mean'] + data['score_std'],
                           alpha=0.2)
        
        plt.title(f'{prob_type} - Target Subproblem Size vs Score')
        plt.xlabel('Target Subproblem Size')
        plt.ylabel('Average Score')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f'{prob_type}_target_comparison.svg'))
        plt.close()

def generate_summary_table(results: Dict[Tuple[str, int], dict]) -> pd.DataFrame:
    """Generate a summary table."""
    rows = []
    for (prob_type, size), data in results.items():
        rows.append({
            'problem_type': prob_type,
            'problem_size': size,
            'best_target_size': data['best_target'],
            'best_score': data['best_score'],
            'avg_subproblems': data['avg_subproblems'],
            'avg_time': data['avg_time'],
            'num_instances': data['num_instances']
        })
    
    return pd.DataFrame(rows)

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Analyzing file: {args.input_csv}")
    
    # Read the CSV file
    df = pd.read_csv(args.input_csv)
    
    # Convert 'inf' to float('inf')
    df['score'] = df['score'].replace('inf', float('inf'))
    
    # Analyze the results
    results = analyze_results(df)
    
    # Generate the summary table
    summary_df = generate_summary_table(results)
    
    # Save the summary table
    summary_path = os.path.join(args.output_dir, 'adaptive_target_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Generate the detailed Markdown report
    report_path = os.path.join(args.output_dir, 'adaptive_target_analysis.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Adaptive Target Size Analysis Report\n\n")
        
        # Add the overall best configuration table
        f.write("## Best Target Size Configuration\n\n")
        f.write("| Problem Type | Problem Size | Best Target Size | Average Score | Average Subproblems | Average Time (seconds) |\n")
        f.write("|--------------|--------------|-----------------|---------------|--------------------|------------------------|\n")
        
        # Sort by problem type and size
        sorted_results = sorted(results.items(), key=lambda x: (x[0][0], x[0][1]))
        for (prob_type, size), data in sorted_results:
            f.write(f"| {prob_type} | {size} | {data['best_target']} | {data['best_score']:.4f} | "
                   f"{data['avg_subproblems']:.2f} | {data['avg_time']:.2f} |\n")
        
        # Add the suggested code update
        f.write("\n## Suggested Code Update\n\n")
        f.write("Based on the analysis above, update the adaptive target size logic in `partitioner_solver_utils.py` as follows:\n\n")
        f.write("```python\n")
        f.write("def get_adaptive_target_size(problem_type: str, problem_size: int) -> int:\n")
        f.write("    \"\"\"Return the best target node count based on problem type and size.\"\n")
        f.write("    targets = {\n")
        
        # Organize the best targets by problem type
        for prob_type in sorted(set(k[0] for k in results.keys())):
            f.write(f"        '{prob_type}': {{\n")
            for size in sorted(size for pt, size in results.keys() if pt == prob_type):
                best_target = results[(prob_type, size)]['best_target']
                f.write(f"            {size}: {best_target},\n")
            f.write("        },\n")
        
        f.write("    }\n")
        f.write("    return targets.get(problem_type, {}).get(problem_size, 50)  # Default value 50\n")
        f.write("```\n")
    
    # Generate the visualization comparison plot
    plot_target_comparison(results, args.output_dir)
    
    print(f"\nAnalysis completed!")
    print(f"- Summary table saved to: {summary_path}")
    print(f"- Detailed report saved to: {report_path}")
    print(f"- Visual comparison figures saved to: {args.output_dir}/*.svg")
    
    # Print key findings
    print("\nKey findings:")
    for prob_type in sorted(set(k[0] for k in results.keys())):
        print(f"\n{prob_type}:")
        for size in sorted(size for pt, size in results.keys() if pt == prob_type):
            data = results[(prob_type, size)]
            print(f"  N={size:4d}: Best target={data['best_target']:3d}, "
                  f"Average score={data['best_score']:.4f}, "
                  f"Average subproblems={data['avg_subproblems']:.2f}")

if __name__ == "__main__":
    main() 