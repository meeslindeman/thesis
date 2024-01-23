import pandas as pd
import os
from datetime import datetime

def get_final_accuracies(directory: str, save: bool = False):
    directory = os.path.abspath(directory)
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    combined_results = pd.DataFrame()

    for file in csv_files:
        filepath = os.path.join(directory, file)
        df = pd.read_csv(filepath)

        # Assuming the last two rows are the final test and train data
        df_final = df.tail(2)

        highest_acc = df['acc'].max()

        # Columns to retrieve. Check if 'topsim' exists in the dataframe
        columns = ['mode', 'acc', 'game_size',  'max_len', 'vocab_size', 'agent', 'embedding_size', 'heads', 'hidden_size']
        if 'topsim' in df.columns:
            columns.append('topsim')

        df_final = df_final[columns].reset_index(drop=True)
        df_final['highest_acc'] = highest_acc

        combined_results = pd.concat([combined_results, df_final], ignore_index=True)

    if save:
        save_directory = os.path.join(directory, 'table')
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_directory, f'final_accuracies_{timestamp}.csv')
        combined_results.to_csv(save_path, index=False)
    else:
        print(combined_results)