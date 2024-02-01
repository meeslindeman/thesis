import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

def plot_metric(folders_dict, maxlen, plot_type, metric='acc', smooth=5, save=False, plot_std=False):
    files = ["dual_2", "dual_3", "dual_4"]
    
    # Initialize subplots
    fig = make_subplots(rows=len(maxlen), 
                        cols=3, 
                        shared_xaxes=True,
                        shared_yaxes=True,
                        column_titles=["Game Size: 4", "Game Size: 10", "Game Size: 22"],
                        row_titles=[f"Max Length: {x}" for x in maxlen],
                        horizontal_spacing=0.02, 
                        vertical_spacing=0.02,
                        x_title="Epochs",
                        y_title="Accuracy" if metric == "acc" else "Message Entropy (bits)")

    # Define colors for train and test
    colors = {'train': 'blue', 'test': 'red' if metric == "acc" else 'darkred'}
    fill_colors = {'train': 'lightblue', 'test': 'pink' if metric == "acc" else 'red'}

    for row, maxlen in enumerate(maxlen, start=1):
        for i, file in enumerate(files, 1):
            all_train_metrics = []
            all_test_metrics = []

            for folder in folders_dict[maxlen]:
                file_path = f"{folder}/{file}.csv"  # Adjust for your file format
                df = pd.read_csv(file_path)

                if plot_type in ['train', 'both']:
                    train_metric = df[df['mode'] == 'train'][metric]
                    all_train_metrics.append(train_metric.rolling(window=smooth, min_periods=1).mean())
                if plot_type in ['test', 'both']:
                    test_metric = df[df['mode'] == 'test'][metric]
                    all_test_metrics.append(test_metric.rolling(window=smooth, min_periods=1).mean())

            # Plotting for the current subplot
            epochs = df['epoch'].unique()
            if plot_type in ['train', 'both']:
                mean_train_metric = pd.concat(all_train_metrics, axis=1).mean(axis=1)
                std_train_deviation = pd.concat(all_train_metrics, axis=1).std(axis=1)

                # Clipping the values for train
                upper_train = np.clip(mean_train_metric + std_train_deviation, 0, 1)
                lower_train = np.clip(mean_train_metric - std_train_deviation, 0, 1)

                fig.add_trace(go.Scatter(x=epochs, y=mean_train_metric, mode='lines', line=dict(color=colors['train']), name=f'Train' if row == 1 and i == 1 else None, showlegend=row == 1 and i == 1), row=row, col=i)
                if plot_std:
                    fig.add_trace(go.Scatter(x=epochs, y=upper_train, mode='lines', line=dict(width=0,color=fill_colors['train']), showlegend=False), row=row, col=i)
                    fig.add_trace(go.Scatter(x=epochs, y=lower_train, mode='lines', line=dict(width=0,color=fill_colors['train']), fill='tonexty', showlegend=False), row=row, col=i)

            if plot_type in ['test', 'both']:
                mean_test_metric = pd.concat(all_test_metrics, axis=1).mean(axis=1)
                std_test_deviation = pd.concat(all_test_metrics, axis=1).std(axis=1)

                # Clipping the values for test
                upper_test = np.clip(mean_test_metric + std_test_deviation, 0, 1)
                lower_test = np.clip(mean_test_metric - std_test_deviation, 0, 1)

                fig.add_trace(go.Scatter(x=epochs, y=mean_test_metric, mode='lines', line=dict(color=colors['test']), name=f'Test' if row == 1 and i == 1 else None, showlegend=row == 1 and i == 1), row=row, col=i)
                if plot_std:
                    fig.add_trace(go.Scatter(x=epochs, y=upper_test, mode='lines', line=dict(width=0,color=fill_colors['test']), showlegend=False), row=row, col=i)
                    fig.add_trace(go.Scatter(x=epochs, y=lower_test, mode='lines', line=dict(width=0,color=fill_colors['test']), fill='tonexty', showlegend=False), row=row, col=i)

            if metric == "acc":
                size = df["game_size"].iloc[0]
                fig.add_hline(y=(1/size), line_dash="dash", line_color="grey", row=row, col=i, annotation_text="Chance Level")
    
    # Update layout to set the range of x-axis from 0 to 300
    x_axis_ranges = {f"xaxis{i}": dict(range=[0, 300]) for i in range(1, len(files) + 1)}
    fig.update_layout(
        height=600 * 4,
        width=1000 * 3,
        **x_axis_ranges,
        legend=dict(orientation="h",
                    xanchor="left",
                    yanchor="bottom",
                    x=0,
                    y=1,
                    font=dict(size=25)
                    ))
    
    fig.update_yaxes(range=[0, 1.0], tickfont=dict(size=18))
    fig.update_xaxes(tickfont=dict(size=18))
    
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 25

    if save:
        fig.write_image(f"plots/{metric}.png", scale=3) 
    else:
        fig.show()

maxlen = [3, 4, 5, 10]
folders_dict = {
    3: [
        "results/maxlen=3/rnd_seed=0",
        "results/maxlen=3/rnd_seed=7",
        "results/maxlen=3/rnd_seed=42"
    ],
    4: [
        "results/maxlen=4/rnd_seed=0",
        "results/maxlen=4/rnd_seed=7",
        "results/maxlen=4/rnd_seed=42"
    ],
    5: [
        "results/maxlen=5/rnd_seed=0",
        "results/maxlen=5/rnd_seed=7",
        "results/maxlen=5/rnd_seed=42"
    ],
    10: [
        "results/maxlen=10/rnd_seed=0",
        "results/maxlen=10/rnd_seed=7",
        "results/maxlen=10/rnd_seed=42"
    ]
}

plot_metric(folders_dict, maxlen, "test", metric="topsim", smooth=3, save=True, plot_std=True)  
