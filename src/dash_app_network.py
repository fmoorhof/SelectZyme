"""This is a minimal example dash app to visualize a networkx graph. Node connections are displayed and the property as hover over event. no legend, no hover possible
Tutorial taken from: https://plotly.com/python/network-graphs/

Todo: think about what to show in the network plot.
edge weights euclidean embidding distance to closest neighbor good?

color scheme and label could be:
- amount of edges: example not ideal
- Label Propagation Algorithm
from networkx.algorithms.community import label_propagation_communities
communities = list(label_propagation_communities(mst_nx))
- Spectral Clustering on Graph Laplacian (downside: requires SKlearn)
...

2. Insights into Connectivity and Routes

    Shortest Paths, Betweenness Centrality, critical nodes
"""
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import networkx as nx
import pandas as pd


def modify_graph_data(G: nx.Graph):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))  # Add node identifier for hover

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
            ),
            line_width=2
        )
    )

    node_adjacencies = [len(list(G.adj[node])) for node in G.nodes()]
    node_trace.marker.color = node_adjacencies

    return edge_trace, node_trace


def run_dash_app(G, df, app: dash.Dash):
    # Modify graph data for visualization
    edge_trace, node_trace = modify_graph_data(G)

    # Create initial figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Minimal Spanning Tree with Objective xy Coloring",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
    )

    # Define app layout
    app.layout = html.Div([
        # Graph plot
        dcc.Graph(id='network-plot', figure=fig),
        
        # Table to show selected node data
        dash_table.DataTable(
            id='data-table',
            columns=[{'id': col, 'name': col} for col in df.columns],
            data=[],  # Start with an empty table
            style_cell={'textAlign': 'left'},
            editable=False,
            row_deletable=False,
            export_format='xlsx',
            export_headers='display',
            merge_duplicate_headers=True,
        )
    ])

    # Callback to update the table based on selected nodes
    @app.callback(
        Output('data-table', 'data'),
        Input('network-plot', 'clickData')
    )
    def update_table(clickData):
        if clickData is None:
            return []

        # Get the selected node
        selected_node = int(clickData['points'][0]['text'])  # Node identifier as an integer

        # Find the corresponding row in the DataFrame
        row = df[df['node_id'] == selected_node]

        # Return the row as a list of dictionaries for the table
        return row.to_dict('records')

    return app



if __name__ == '__main__':
    # Generate minimal example data (network)
    G = nx.random_geometric_graph(10, 0.5, seed=42)
    pos = nx.spring_layout(G)  # Generate positions for visualization
    nx.set_node_attributes(G, pos, 'pos')  # Assign positions as attributes
    G = nx.minimum_spanning_tree(G)

    # Create a DataFrame with minimal node information
    df = pd.DataFrame({
        'node_id': list(G.nodes()),  # Node identifiers
        'x': [G.nodes[node]['pos'][0] for node in G.nodes()],  # x-coordinates
        'y': [G.nodes[node]['pos'][1] for node in G.nodes()],  # y-coordinates
        '# connections': [len(list(G.adj[node])) for node in G.nodes()]  # Number of connections
    })


    app = dash.Dash(__name__)
    run_dash_app(G, df, app).run_server(debug=True)
