import os
import pandas as pd
import plotly.graph_objs as go
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

def plot_entropy(base_folder, plot, plot_std=False, save=False):
    df = load_all_data(base_folder)

    # Filter the dataframe to include only rows with high accuracy
    high_accuracy_df = df[df['epoch'] == 300]

    # Calculate the mean and standard deviation of message entropy for each acc and game_size
    stats_df = high_accuracy_df.groupby(['max_len', 'game_size']).agg(
        mean_entropy=(plot, 'mean'),
        std_entropy=(plot, lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
    ).reset_index()

    fig = go.Figure()
    colors = {3: px.colors.qualitative.D3[0], 4: px.colors.qualitative.D3[1], 5: px.colors.qualitative.D3[2], 10: px.colors.qualitative.D3[3]}

    # Ensure data is ordered correctly for plotting
    stats_df = stats_df.sort_values(by=['max_len', 'game_size'])

    for max_len in stats_df['max_len'].unique():
        subset = stats_df[stats_df['max_len'] == max_len]
        color = colors[max_len]

        # Plotting the mean with lines and markers for each max_len across all game sizes
        fig.add_trace(go.Scatter(
            x=subset['game_size'], y=subset['mean_entropy'],
            mode='lines+markers', name=f'Max Len: {max_len}',
            line=dict(color=color, width=4),
            marker=dict(size=8, line=dict(width=.5)),
            showlegend=True
        ))

        if plot_std:
            # Upper bound of the standard deviation
            fig.add_trace(go.Scatter(
                x=subset['game_size'], y=subset['mean_entropy'] + subset['std_entropy'],
                mode='lines', name=f'Upper Bound Max Len: {max_len}',
                line=dict(color=color, width=0),
                showlegend=False
            ))

            # Lower bound of the standard deviation with fill to upper bound
            fig.add_trace(go.Scatter(
                x=subset['game_size'], y=subset['mean_entropy'] - subset['std_entropy'],
                mode='lines', name=f'Lower Bound Max Len: {max_len}',
                line=dict(color=color, width=0),
                fill='tonexty', opacity=0.1,
                showlegend=False
            ))
    
    unique_game_sizes = stats_df['game_size'].unique()
    log_values = np.log2(unique_game_sizes)

    # Plot log_2(N) as a grey dotted line
    fig.add_trace(go.Scatter(
        x=unique_game_sizes, y=log_values,
        mode='lines', name='Hmin',
        line=dict(color='grey', dash='dot'),
        showlegend=True
    ))

    fig.update_xaxes(type='log')

    fig.update_layout(height=600,
                      width=1000,
                      xaxis_title='Game Size (log base-2)',
                      yaxis_title='Mean Message Entropy (bits)',
                      legend=dict(
                          orientation="h",
                          xanchor="left",
                          yanchor="bottom",
                          x=0,
                          y=1))

    if save:
        fig.write_image("plots/entropy.png", scale=3)
    else:
        fig.show()


plot_entropy("results", plot="message_entropy", plot_std=True, save=True)