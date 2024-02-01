import os
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from collections import Counter
import ast

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

def plot_message_histogram(base_folder, mode='both', cap=50, save=False):
    df = load_all_data(base_folder)
    df = df.dropna(subset=['messages'])

    df = df[df['game_size'] != 10]

    # Filter based on the mode
    if mode in ['train', 'test']:
        df = df[df['mode'] == mode]

    df['messages'] = df['messages'].apply(ast.literal_eval)

    game_sizes = df['game_size'].unique()
    fig = make_subplots(rows=1,
                        cols=len(game_sizes), 
                        subplot_titles=[f"Game Size: {x}" for x in sorted(game_sizes)],
                        x_title='Message Sequence',
                        shared_yaxes=True,
                        horizontal_spacing=0.02)

    for i, game_size in enumerate(sorted(game_sizes), start=1):
        # Filter dataframe for the current game size
        game_size_df = df[df['game_size'] == game_size]

        # Create a Counter object to store the frequency of each message sequence
        sequence_counts = Counter()

        # Iterate over each row, then iterate over each message sequence in the 'messages' column
        for index, row in game_size_df.iterrows():
            for sequence in row['messages']:
                sequence_counts[tuple(sequence)] += 1

        histogram_data = pd.DataFrame({
            'Message Sequence': [str(list(seq)) for seq in sequence_counts.keys()],
            'Frequency': list(sequence_counts.values())
        })

        histogram_data = histogram_data.sort_values(by='Frequency', ascending=False).head(cap)
        
        # Use bar instead of histogram for custom data
        fig.add_trace(
            go.Bar(x=histogram_data['Message Sequence'], y=histogram_data['Frequency'], name=f"Game Size: {game_size}"),
            row=1, col=i
        )

    fig.update_layout(height=600, width=1000 * len(game_sizes),
                      yaxis_title='Message Frequency')
    
    if save:
        fig.write_image(f"plots/message_histogram.png", scale=3)
    else:
        fig.show()

plot_message_histogram("results/maxlen=5/rnd_seed=7", mode="test", cap=5, save=True)