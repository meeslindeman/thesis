import pandas as pd 
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import ast
import numpy as np

def load_all_data(base_folder):
    all_data = []
    for subdir, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path)
                all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def plot_life(base_folder, n, mode="test", save=False):
    df = load_all_data(base_folder)
    df = df.dropna(subset=['messages'])

    df = df[df['game_size'] != 10]
    
    # Filter based on the mode
    if mode in ['train', 'test']:
        df = df[df['mode'] == mode]

    df["messages"] = df["messages"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    colors_1 = px.colors.qualitative.Set1
    colors_2 = px.colors.qualitative.Bold
    
    game_sizes = df['game_size'].unique()
    fig = make_subplots(rows=1,
                        cols=len(game_sizes), 
                        subplot_titles=[f"Game Size: {x}" for x in sorted(game_sizes)],
                        shared_yaxes=False,
                        horizontal_spacing=0.018,
                        x_title="Epochs",
                        y_title="Message Frequency")

    for i, game_size in enumerate(sorted(game_sizes), start=1):
        if i == 1:
            pallette = colors_1
        else:
            pallette = colors_2

        # Filter dataframe for the current game size
        game_size_df = df[df['game_size'] == game_size]

        sub_df = game_size_df.tail(50)

        # Count the frequency of each unique message across all epochs
        all_messages = [tuple(message) for sublist in sub_df['messages'].tolist() for message in sublist]
        message_counts = Counter(all_messages)

        # Get the top n messages based on their frequency
        top_messages = [msg for msg, _ in message_counts.most_common(n)]

        # Initialize a dictionary to track the life of each top message
        message_life = {message: [0] * (df['epoch'].max() + 1) for message in top_messages}

        # Populate the message_life with the cumulative frequency of each message for each epoch
        for epoch in range(0, df['epoch'].max() + 1):
            epoch_messages = game_size_df[game_size_df['epoch'] == epoch]['messages'].tolist()
            for message in top_messages:
                current_count = sum(map(lambda m: m.count(list(message)), epoch_messages))
                message_life[message][epoch] = current_count

        # Now calculate the cumulative sum for the entire life span of each message
        for message in top_messages:
            message_life[message] = np.cumsum(message_life[message])

        color_index = 0
        # Plot the life of each top message
        for message, life_counts in message_life.items():
            color = pallette[color_index % len(pallette)]
            color_index += 1

            # Cumulative sum to get the life of the message
            life_counts = np.cumsum(life_counts)  

            # Find the last index where the count increases
            last_increase_idx = np.where(np.diff(life_counts) > 0)[0][-1] + 1 if np.diff(life_counts).any() else 0
            life_counts = life_counts[:last_increase_idx + 1]
            epochs = list(range(0, last_increase_idx + 2))

            fig.add_trace(
                go.Scatter(
                    x=epochs, 
                    y=life_counts, 
                    mode='lines+markers', 
                    marker=dict(size=2), line=dict(width=3, color=color),
                    name=f'{message}'
                ), 
                row=1, col=i
            )

    fig.update_layout(height=600,
                      width=1000 * len(game_sizes))

    if save:
        fig.write_image("plots/message_lifespan_2.png")
    else:
        fig.show()

plot_life("results/maxlen=4/rnd_seed=0", n=10, mode="test", save=True)