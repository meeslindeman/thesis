import pandas as pd
import os

# Function to load all CSV files from the nested folder structure
def load_all_data(base_folder):
    all_data = []
    for subdir, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path)
                all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def calculate_final_metrics(base_folder):
    df = load_all_data(base_folder)

    # Filter relevant columns
    relevant_columns = ['mode', 'epoch', 'acc', 'loss', 'topsim', 'message_entropy', 'game_size', 'max_len', 'random_seed']
    df_filtered = df[relevant_columns]
    
    # Filter out to the last epoch of each mode (assuming 300 epochs for train and test)
    df_last_epoch = df_filtered[df_filtered['epoch'] == 300]
    
    # Define the aggregation dictionary for the metrics
    aggregation_metrics = {
        'acc': 'mean',
        'topsim': 'mean',
        'message_entropy': 'mean',
    }
    
    # Group by 'max_len', 'game_size', 'mode' and aggregate using the defined metrics
    final_metrics_df = df_last_epoch.groupby(['max_len', 'game_size', 'mode']).agg(aggregation_metrics).reset_index()

    final_metrics_df.to_csv('results/final.csv', index=False)

calculate_final_metrics('results')

