import dash
from dash import dcc, html, Output, Input
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
import json
import numpy as np
from constants import BOARD_RANGE, STRATEGY, DASH_PORT, USE_EPSILON, VIEW_DATA_WINDOW, EVALUATE_MODEL

episode_fitness = []
episode_steps = []
episode_epsilon = []
episode_snake_length = []
session_completions = 0
episode_count = 0

app = dash.Dash(__name__)

app.layout = html.Div([
  dcc.Graph(id='live-graph', animate=False),
  dcc.Interval(id='interval', interval=1000, n_intervals=0),
  html.Div([
    dcc.Checklist(
      id='pause-switch',
      options=[{'label': ' Pause Updates', 'value': 'pause'}],
      value=[],
      inline=True,
      style={'fontSize': 20, 'marginBottom': '20px'}
    )
  ])
])

@app.callback(
  Output('live-graph', 'figure'),
  [Input('interval', 'n_intervals'), Input("pause-switch", "value")]
)
def update_graph(n, pause_value):
  # If the "Pause" checkbox is checked, stop updating the graph
  if 'pause' in pause_value:
    raise dash.exceptions.PreventUpdate

  # Calculate averages for each metric
  avg_steps = np.mean(episode_steps) if len(episode_steps) > 0 else 0
  avg_length = np.mean(episode_snake_length) if len(episode_snake_length) > 0 else 0
  avg_fitness = np.mean(episode_fitness) if len(episode_fitness) > 0 else 0

  display_episodes = list(range(len(episode_steps)))

  if STRATEGY == "dqn":
    # If there's no data yet, return an empty figure
    if not episode_fitness:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=3,
    )

    fig.add_trace(
      go.Scatter(
        x=display_episodes,
        y=episode_fitness,
        mode='lines+markers',
        name='Fitness',
        line=dict(dash='solid', color='darkblue')
      ),
      row=1, col=3
    )

    fig.add_trace(
      go.Scatter(
        x=display_episodes,
        y=[avg_fitness] * len(episode_fitness),
        mode='lines',
        name='Avg Fitness',
        line=dict(dash='dash', color='red')
      ),
      row=1, col=3
    )

    fig.add_trace(
      go.Scatter(
        x=display_episodes,
        y=episode_steps,
        mode='lines+markers',
        name='Steps',
        line=dict(dash='solid', color='orange'),
      ),
      row=1, col=2
    )

    fig.add_trace(
      go.Scatter(
        x=display_episodes,
        y=[avg_steps] * len(episode_steps),
        mode='lines',
        name='Avg Steps',
        line=dict(dash='dash', color='red')
      ),
      row=1, col=2
    )

    if USE_EPSILON:
      fig.add_trace(
        go.Scatter(
          x=display_episodes,
          y=episode_epsilon,
          mode='lines+markers',
          name='Epsilon',
          line=dict(dash='solid', color='mediumvioletred'),
        ),
        row=2, col=1
      )

    fig.add_trace(
      go.Scatter(
        x=display_episodes,
        y=episode_snake_length,
        mode='lines+markers',
        name='Snake Length',
        line=dict(dash='solid', color='lightpink'),
      ),
      row=1, col=1
    )

    fig.add_trace(
      go.Scatter(
        x=display_episodes,
        y=[avg_length] * len(episode_snake_length),
        mode='lines',
        name='Avg Snake Length',
        line=dict(dash='dash', color='red')
      ),
      row=1, col=1
    )

    fig.update_layout(uirevision=str(n), height=800, width=2000)

    return fig
  
  else:
    if not episode_snake_length:
      return go.Figure()

    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(
      go.Scatter(
        x=display_episodes, y=episode_steps,
        mode='lines+markers',
        name='Steps',
        line=dict(dash='solid', color='orange')
      ), row=1, col=2
    )

    fig.add_trace(
      go.Scatter(
        x=display_episodes, y=[avg_steps] * len(episode_steps),
        mode='lines',
        name='Avg Steps',
        line=dict(dash='dash', color='red')
      ), row=1, col=2
    )

    fig.add_trace(
      go.Scatter(
        x=display_episodes, y=episode_snake_length,
        mode='lines+markers',
        name='Size Ratio',
        line=dict(dash='solid', color='lightblue')
      ), row=1, col=1
    )
    
    fig.add_trace(
      go.Scatter(
        x=display_episodes, y=[avg_length] * len(episode_snake_length),
        mode='lines',
        name='Avg Size Ratio',
        line=dict(dash='dash', color='red')
      ), row=1, col=1
    )

    # Update layout
    fig.update_layout(uirevision=str(n), height=800, width=2000)

    return fig

def run_dash():
    app.run_server(debug=False, use_reloader=False, port=DASH_PORT)

def save_data(data, file_name):
  if file_name is not None:
    with open(file_name, 'a') as f:
      f.write(data)
      f.flush()

def load_data(file_name):
  if file_name is not None:
    with open(file_name, 'r') as f:
      return json.load(f)
  else:
    print(f"Failed to load file: {file_name}")
