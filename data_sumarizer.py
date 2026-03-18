import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

output_folder = "training_benchmarks"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")

def extract_summary_data(log_dir):
    """Extracts scalar data and merges them into a single robust DataFrame."""
    ea = event_accumulator.EventAccumulator(log_dir,
        size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    tags = ea.Tags()['scalars']
    combined_df = None

    for tag in tags:
        events = ea.Scalars(tag)
        # Create a dataframe for this specific tag
        df_tag = pd.DataFrame([{'step': e.step, tag: e.value} for e in events])
        df_tag.set_index('step', inplace=True)

        if combined_df is None:
            combined_df = df_tag
        else:
            # Join dataframes on the 'step' index
            combined_df = combined_df.join(df_tag, how='outer')
            
    return combined_df

def process_all_experiments(base_runs_dir):
    # Mapping folders to their display names
    scenarios = {
        #'air3d_run': 'Scenario 1 (3D)',
        'collision_6d_run': 'Scenario 2 (6D)',
        'collision_9d_run': 'Scenario 3 (9D)',
        'narrow_passage_10d_run': 'Scenario 4 (10D)'
    }
    
    all_results = {}

    for folder, name in scenarios.items():
        summary_path = os.path.join(base_runs_dir, folder, 'training', 'summaries')
        
        # Check for event files
        if not os.path.exists(summary_path):
            print(f"Skipping {name}: Path not found.")
            continue

        print(f"Processing {name}...")
        df = extract_summary_data(summary_path)
        
        if df is not None:
            # Save the clean CSV
            csv_filename = os.path.join(output_folder, f"{folder}_full_benchmarks.csv")
            df.to_csv(csv_filename)
            all_results[folder] = df
            metrics_mapping = {
                'total_train_loss': 'Total Training Loss',
                'diff_constraint_hom': 'PDE Constraint (Gradient Accuracy)',
                'dirichlet': 'Boundary (Dirichlet) Loss'
            }
            # Find which of these actual headers exist in the dataframe
            metrics_found = [m for m in metrics_mapping.keys() if m in df.columns]
            print(f"Metrics found for {name}: {metrics_found}")

            for metric in metrics_found:
                plt.figure(figsize=(10, 5))
                # Dropna is important because metrics are logged at different intervals
                df[metric].dropna().plot()
                
                plt.title(f"{name} - {metrics_mapping[metric]}")
                plt.xlabel("Iteration")
                plt.ylabel("Value")
                plt.grid(True)
                
                # Save with a clean filename
                file_tag = metric.replace('_', '')
                plot_filename = os.path.join(output_folder, f"{folder}_{file_tag}_plot.png")
                plt.savefig(plot_filename)
                print(f"Saved: {folder}_{file_tag}_plot.png")
                plt.close()
        else:
            print(f"No data found in {folder}")

    return all_results

if __name__ == "__main__":
    runs_directory = './runs' 
    process_all_experiments(runs_directory)
    print("\nExtraction complete. Check CSVs for metrics and PNGs for trends.")