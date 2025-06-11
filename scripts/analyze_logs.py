import os
import glob
from collections import defaultdict
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_tensorboard_logs(log_dir):
    """Load all tensorboard logs from the specified directory."""
    runs = {}
    for run_dir in glob.glob(os.path.join(log_dir, "PPO_*")):
        run_name = os.path.basename(run_dir)
        try:
            ea = event_accumulator.EventAccumulator(
                run_dir,
                size_guidance={
                    event_accumulator.SCALARS: 0,
                }
            )
            ea.Reload()
            
            # Get all scalar tags
            tags = ea.Tags()['scalars']
            run_data = {}
            
            for tag in tags:
                events = ea.Scalars(tag)
                run_data[tag] = {
                    'steps': [event.step for event in events],
                    'values': [event.value for event in events],
                    'wall_times': [event.wall_time for event in events]
                }
            
            runs[run_name] = run_data
        except Exception as e:
            print(f"Error loading {run_name}: {str(e)}")
            continue
    
    return runs

def analyze_runs(runs):
    """Analyze the runs and return key statistics."""
    analysis = defaultdict(dict)
    
    for run_name, run_data in runs.items():
        # Get the final episode reward
        if 'episode/reward' in run_data:
            rewards = run_data['episode/reward']['values']
            analysis[run_name]['final_reward'] = rewards[-1]
            analysis[run_name]['max_reward'] = max(rewards)
            analysis[run_name]['mean_reward'] = np.mean(rewards)
            analysis[run_name]['std_reward'] = np.std(rewards)
            
            # Calculate convergence metrics
            window_size = min(100, len(rewards))
            if len(rewards) >= window_size:
                final_window_mean = np.mean(rewards[-window_size:])
                analysis[run_name]['convergence'] = final_window_mean
        
        # Get episode lengths if available
        if 'episode/length' in run_data:
            lengths = run_data['episode/length']['values']
            analysis[run_name]['mean_length'] = np.mean(lengths)
            analysis[run_name]['std_length'] = np.std(lengths)
        
        # Get any other metrics
        for tag, data in run_data.items():
            if tag not in ['episode/reward', 'episode/length']:
                values = data['values']
                analysis[run_name][f'{tag}_mean'] = np.mean(values)
                analysis[run_name][f'{tag}_std'] = np.std(values)
    
    return analysis

def plot_learning_curves(runs, metric='episode/reward', save_path=None):
    """Plot learning curves for all runs."""
    plt.figure(figsize=(12, 6))
    
    for run_name, run_data in runs.items():
        if metric in run_data:
            steps = run_data[metric]['steps']
            values = run_data[metric]['values']
            plt.plot(steps, values, label=run_name, alpha=0.7)
    
    plt.title(f'Learning Curves - {metric}')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    # Load and analyze logs
    log_dir = "tensorboard_logs"
    runs = load_tensorboard_logs(log_dir)
    
    if not runs:
        print("No valid runs found in the log directory!")
        return
        
    analysis = analyze_runs(runs)
    
    # Convert analysis to DataFrame for better viewing
    df = pd.DataFrame.from_dict(analysis, orient='index')
    
    # Sort by final reward if available, otherwise by mean reward
    sort_column = 'final_reward' if 'final_reward' in df.columns else 'mean_reward'
    if sort_column in df.columns:
        df = df.sort_values(sort_column, ascending=False)
    
    # Print analysis results
    print("\nRun Analysis Summary:")
    print("=" * 80)
    print(df.round(2))
    
    # Create output directory for plots
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots for all available metrics
    for metric in ['episode/reward', 'episode/length']:
        if any(metric in run_data for run_data in runs.values()):
            plot_learning_curves(runs, metric, 
                               save_path=output_dir / f'{metric.replace("/", "_")}_curves.png')
    
    # Save analysis to CSV
    df.to_csv(output_dir / 'run_analysis.csv')
    
    print(f"\nAnalysis results saved to {output_dir}")
    
    # Print summary of best runs
    if 'final_reward' in df.columns:
        print("\nBest performing runs:")
        print("=" * 80)
        best_runs = df.nlargest(5, 'final_reward')
        print(best_runs[['final_reward', 'max_reward', 'mean_reward']].round(2))

if __name__ == "__main__":
    main() 