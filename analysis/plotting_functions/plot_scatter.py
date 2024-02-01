import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Function to load all CSV files from the nested folder structure
def load_all_data(base_folder):
    all_data = []
    for subdir, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path)
                df = df.tail(40)
                all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def plot_scatter(base_folder, x_axis, y_axis, hue='game_size', save=True):
    df = load_all_data(base_folder)

    # Convert game_size to string for categorical color
    df["game_size"] = df["game_size"].astype(str)

    # Get unique max_len values
    max_lens = df['max_len'].unique()
    max_lens.sort()

    titles = {"acc": "Accuracy", "message_entropy": "Mean Message Entropy (bits)", "topsim": "Mean TopSim"}
    if y_axis == "topsim":
        game_size_colors = {"4": px.colors.qualitative.Vivid[0], "10": px.colors.qualitative.Vivid[1], "22": px.colors.qualitative.Vivid[2]}
    elif y_axis == "message_entropy":
        game_size_colors = {"4": px.colors.qualitative.Dark2[0], "10": px.colors.qualitative.Dark2[1], "22": px.colors.qualitative.Dark2[2]}

    df = df.groupby(['acc', 'game_size', 'max_len']).agg(
        metric_x=(x_axis, 'mean'),
        metric_y=(y_axis, 'mean')).reset_index()
    
    # Create a subplot for each max_len value
    fig = make_subplots(rows=2, 
                        cols=2, 
                        shared_xaxes=True, 
                        shared_yaxes=True,  
                        subplot_titles=[f"Max Length: {x}" for x in max_lens],
                        vertical_spacing=0.03,
                        horizontal_spacing=0.02,
                        x_title=titles[x_axis],
                        y_title=titles[y_axis])

    for i, max_len in enumerate(max_lens, 1):
        # Calculate row and column
        row = (i - 1) // 2 + 1
        col = (i - 1) % 2 + 1

        # Filter the dataframe for each max_len
        filtered_df = df[df['max_len'] == max_len]

        for game_size in filtered_df['game_size'].unique():
            sub_df = filtered_df[filtered_df['game_size'] == game_size]
            fig.add_trace(
                go.Scatter(x=sub_df["metric_x"], y=sub_df["metric_y"], mode='markers', name=game_size, marker_color=game_size_colors[game_size], marker=dict(size=8), showlegend=(i == 1)),
                row=row, col=col)

    # Update legend
    fig.update_layout(legend=dict(
            title="Game Size",
            orientation="h",
            xanchor="left",
            yanchor="bottom",
            font=dict(size=20),
            x=0,
            y=1),
            height=300 * len(max_lens),
            width=500 * len(max_lens))
    if y_axis == "message_entropy":
        fig.update_yaxes(range=[0, 10])
    elif y_axis == "topsim":
        fig.update_yaxes(range=[0, 1.0])

    fig.update_yaxes(tickfont=dict(size=20))
    fig.update_xaxes(tickfont=dict(size=20))
    
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 20

    if save:
        fig.write_image(f"plots/{y_axis}_scatter.png", scale=3)
    else:
        fig.show()

plot_scatter('results', x_axis='topsim', y_axis='message_entropy', save=True)


