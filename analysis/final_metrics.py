import pandas as pd
import os

def get_final_accuracies(directory: str):
    directory = os.path.abspath(directory)
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    combined_results = pd.DataFrame()

    for file in csv_files:
        filepath = os.path.join(directory, file)
        df = pd.read_csv(filepath)

        # Assuming the last two rows are the final test and train data
        df_final = df.tail(2)

        # Columns to retrieve. Check if 'topsim' exists in the dataframe
        columns = ['mode', 'acc', 'max_len', 'game_size', 'vocab_size', 'agent']
        if 'topsim' in df.columns:
            columns.append('topsim')

        df_final = df_final[columns].reset_index(drop=True)

        # Add a column to identify the file, if needed
        df_final['source_file'] = file

        combined_results = pd.concat([combined_results, df_final], ignore_index=True)

    return combined_results

# Example usage
final_accuracies = get_final_accuracies("../results")
print(final_accuracies)