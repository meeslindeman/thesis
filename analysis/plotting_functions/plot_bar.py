import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

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

def plot_topsim_barchart(base_folder, save=False):
    df = load_all_data(base_folder)

    # Filter the dataframe to include only rows with high accuracy
    # high_accuracy_threshold = 0.75
    # high_accuracy_df = df[df['acc'] >= high_accuracy_threshold]
    high_accuracy_df = df[df['epoch'] == 300]

    # Group data by 'max_len' and 'game_size' and calculate mean and std of 'topsim'
    grouped = high_accuracy_df.groupby(['max_len', 'game_size']).agg(
        topsim_mean=('topsim', 'mean'),
        topsim_std=('topsim', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
    ).reset_index()

    fig = go.Figure()
    max_lens = grouped['max_len'].unique()
    game_sizes = grouped['game_size'].unique()

    colors = px.colors.qualitative.T10[:len(max_lens)]

    # Bar width configuration
    total_bar_width = 0.5
    single_bar_width = total_bar_width / len(max_lens)

    # Add bars to the figure for each game_size and max_len
    for i, max_len in enumerate(max_lens):
        for j, game_size in enumerate(game_sizes):
            data = grouped[(grouped['max_len'] == max_len) & (grouped['game_size'] == game_size)]
            fig.add_trace(go.Bar(
                x=[game_size],
                y=data['topsim_mean'],
                error_y=dict(type='data', array=data['topsim_std']),
                name=f"Max Length: {max_len}",
                marker_color=colors[i],
                width=single_bar_width,
                offset=(single_bar_width * i) - (total_bar_width / 2),
                showlegend=(j == 0)
            ))

    # Update layout
    fig.update_layout(
        barmode='group',
        height=600,
        width=1000,
        xaxis=dict(type='category'),
        xaxis_title="Game Size",
        yaxis_title="Mean TopSim",
        legend=dict(
          orientation="h",
          xanchor="left",
          yanchor="bottom",
          x=0,
          y=1))

    if save:
        fig.write_image(f"plots/topsim_bar.png", scale=3) 
    else:
        fig.show()

plot_topsim_barchart('results', save=True)
