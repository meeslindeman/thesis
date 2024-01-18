import os
import re
from options import Options
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def load_dataframes_from_folder(folder_path: str) -> list:
    dataframes = []
    filenames = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, file))
            dataframes.append(df)
            filenames.append(file)
    return dataframes, filenames

def plot_all_experiments(folder_path: str, mode='both', save=False):
    colors = ["#FFA500", "#6495ED"]

    dfs, filenames = load_dataframes_from_folder(folder_path)
    if not dfs:
        print("No data found in the specified folder.")
        return

    # Extract options from filename
    option_a = sorted(set(f.split('_')[0] for f in filenames))
    option_b = sorted(set(f.split('_')[1].split('.')[0] for f in filenames))

    # Set the number of rows and columns
    rows = len(option_b)
    cols = len(option_a)

    # Create a subplot
    fig = make_subplots(rows=rows, 
                        cols=cols, 
                        shared_xaxes=True,
                        shared_yaxes=True,
                        column_titles=[f"Vocab size: {a}" for a in option_a],
                        row_titles=[f"Max Len: {b}" for b in option_b], # Modify to option set
                        horizontal_spacing=0.01, 
                        vertical_spacing=0.02,
                        x_title="Epochs",
                        y_title="Accuracy")

    for df, filename in zip(dfs, filenames):
        if mode != 'both':
            df = df[df['mode'] == mode]

        # Add a plot for each DataFrame
        a, b = filename.split('_')
        b = b.split('.')[0]
        col = option_a.index(a) + 1
        row = option_b.index(b) + 1 

        for i, m in enumerate(['train', 'test']):
            sub_df = df[df['mode'] == m]
            color = colors[i]  

            trace = go.Scatter(x=sub_df['epoch'], 
                               y=sub_df['acc'], 
                               mode='lines+markers', 
                               name=f'{m}', 
                               line=dict(color=color),
                               showlegend=True if (row, col) == (1, 1) else False)
            
            fig.add_trace(trace, row=row, col=col)
        fig.update_traces(mode='markers+lines', marker=dict(size=4, line=dict(width=1)), line=dict(width=4))

    # Update layout
    fig.update_layout(legend=dict(
                        orientation="h",
                        xanchor="left",
                        yanchor="bottom",
                        x=0,
                        y=1),
                      height=400 * rows,
                      width=800 * cols)
    
    size = df["game_size"].iloc[0]
    for idx in range(len(dfs)):
        row, col = (idx // cols) + 1, (idx % cols) + 1
        fig.add_hline(y=(1/size), line_dash="dash", line_color="grey", row=row, col=col, annotation_text="Chance Level")

    if save:
        fig.write_image(f"plots/combined_accuracy_plots.png") # Modify to option set
    else:
        fig.show()

def plot_experiment(opts: Options, df: pd.DataFrame, mode='both', save=True):
    colors = ["#FFA500", "#6495ED"]
    size = df["game_size"].iloc[0]

    if mode != 'both':
        df = df[df['mode'] == mode]

    fig = px.line(df, x='epoch', y='acc', color='mode', color_discrete_sequence=colors)
    
    fig.update_traces(mode='markers+lines', marker=dict(size=4, line=dict(width=1)), line=dict(width=4))

    # Add a grey dotted line for chance level
    fig.add_hline(y=(1/size), line_dash="dash", line_color="grey", annotation_text="Chance Level")

    fig.update_xaxes(title_text='Epoch')  # Update x-axis title
    fig.update_yaxes(title_text='Accuracy')  # Update y-axis title

    # Customizing the legend title
    fig.update_layout(legend_title_text='Mode')

    if save:
        # pip3 install kaleido
        fig.write_image(f"plots/{opts.agents}.png")
    else:
        fig.show()