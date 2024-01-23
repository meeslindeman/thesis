import os
import pandas as pd
import plotly.express as px

def plot_average_accuracy(folder_path):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    df_list = [pd.read_csv(file) for file in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    grouped = combined_df.groupby(['agent', 'mode'])['acc'].agg(['mean', 'std']).reset_index()

    fig = px.bar(grouped, 
                 x='agent', 
                 y='mean', 
                 color='mode', 
                 barmode='group',
                 error_y='std', 
                 labels={'mean': 'Average Accuracy', 'game_size': 'Game Size', 'agent': 'Agent'},
                 title='', 
                 color_discrete_sequence=px.colors.qualitative.Plotly, 
                 opacity=0.8)
    
    fig.update_traces(marker_line_width=10)

    fig.update_layout(legend_title_text='Mode', 
                      legend=dict(
                        orientation="h",
                        xanchor="left",
                        yanchor="bottom",
                        x=0,
                        y=1))

    fig.update_yaxes(range=[0, 1.0])
    
    fig.show()

plot_average_accuracy('../results/table/plot')
