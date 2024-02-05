import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
from collections import Counter

def load_all_data(base_folder):
    all_data = []
    for subdir, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path)
                all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def plot_message_frequency(base_folder, n, mode="test", save=False):
    df = load_all_data(base_folder)
    df = df.dropna(subset=['messages'])

    df = df[df['game_size'] != 10]

    # Filter based on the mode
    if mode in ['train', 'test']:
        df = df[df['mode'] == mode]

    df["messages"] = df["messages"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    game_sizes = df['game_size'].unique()
    fig = make_subplots(rows=1, cols=len(game_sizes), 
                        subplot_titles=[f"Game Size: {x}" for x in sorted(game_sizes)],
                        x_title='Epochs',
                        y_title='Message Frequency',
                        horizontal_spacing=0.02,
                        shared_yaxes=False)

    for i, game_size in enumerate(sorted(game_sizes), start=1):
        game_size_df = df[df['game_size'] == game_size]
        epochs = sorted(game_size_df['epoch'].unique())

        sub_df = game_size_df.tail(50)
        
        all_messages = Counter([tuple(msg) for sublist in sub_df['messages'].tolist() for msg in sublist])

        if n:
            top_messages = [msg for msg, _ in all_messages.most_common(n)]
        else:
            top_messages = list(all_messages.keys())

        message_frequencies = {msg: [0] * len(epochs) for msg in top_messages}

        for idx, epoch in enumerate(epochs):
            epoch_messages = game_size_df[game_size_df['epoch'] == epoch]['messages'].tolist()
            epoch_message_counts = Counter([tuple(msg) for sublist in epoch_messages for msg in sublist])

            for msg in top_messages:
                if msg in epoch_message_counts:
                    message_frequencies[msg][idx] += epoch_message_counts[msg]

        # Plotting
        for msg, freqs in message_frequencies.items():
            nonzero_freqs = [(epoch, freq) for epoch, freq in zip(epochs, freqs) if freq > 0]

            if nonzero_freqs:
                x_values, y_values = zip(*nonzero_freqs)  # Unzip the list of tuples
                fig.add_trace(
                    go.Scatter(
                        x=x_values, 
                        y=y_values, 
                        mode='lines',
                        name=f'{msg}',
                        line=dict(width=3),
                    ), 
                    row=1, col=i
                )

    fig.update_layout(height=600, width=1000 * len(game_sizes))
    fig.update_xaxes(range=[0, 300])
    
    if save:
        fig.write_image("plots/message_frequency_n.png")
    else:
        fig.show()

plot_message_frequency("results/maxlen=4/rnd_seed=0", n=10, mode="test", save=True)