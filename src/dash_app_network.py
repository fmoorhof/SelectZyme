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


def modify_graph_data(G, df):
    """Generate edge and node traces for a NetworkX graph with improved hover data."""
    if not isinstance(G, nx.Graph):
        G = G.to_networkx()
    # annotate nodes with df data
    for node in G.nodes():
        if node in df.index:
            nx.set_node_attributes(G, {node: df.loc[node].to_dict()})

    # define graph layout and coordinates
    #   # if not done in visualizer.py hdbscan
    pos = nx.spring_layout(G)
    pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", root=0)  # Warning: specified root node "0" was not found.Using default calculation for root node
    nx.set_node_attributes(G, pos, 'pos')

    # Edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])  # Use extend for cleaner code
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',  # No hover info for edges
        mode='lines'
    )
    
    # Node traces
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

        # # Remove unnecessary attributes
        # G.nodes[node].pop('data', None)  # Remove 'data' if it exists
        # G.nodes[node].pop('pos', None)  # Remove 'pos' if it exists

        # # Format node attributes for hover text
        # hover_text = "<br>".join([f"{key}: {value}" for key, value in G.nodes[node].items()])
        # node_text.append(hover_text)

    columns_of_interest = [col for col in df.columns if col not in ['sequence', 'BRENDA URL', 'lineage', 'marker_size', 'marker_symbol', 'selected', 'organism_id']]
    # columns_of_interest = ['accession', 'reviewed', 'ec', 'length', 'xref_brenda', 'xref_pdb', 'cluster', 'species', 'domain', 'kingdom', 'selected']
    node_text=["<br>".join(f"{col}: {df[col][i]}" for col in columns_of_interest)
            for i in range(len(df))]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        hoverinfo='text',
        customdata=df['accession'],
        text=node_text,  # Use formatted hover text
        mode='markers',
        marker=dict(
            size=df['marker_size'],
            symbol=df['marker_symbol'],
            opacity=0.7,
            line_width=1,

            # connectivity legend
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],  # Will be populated with node adjacencies
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
            )
        )
    )

    # Color nodes by their number of connections
    node_adjacencies = [len(list(G.adj[node])) for node in G.nodes()]
    node_trace.marker.color = node_adjacencies

    return edge_trace, node_trace


def run_dash_app(G, df, app: dash.Dash, fig):
    # Modify graph data for visualization
    #edge_trace, node_trace = modify_graph_data(G)

    # Create initial figure
    # fig = go.Figure(
    #     data=[edge_trace, node_trace],
    #     layout=go.Layout(
    #         title="Minimal Spanning Tree with Objective xy Coloring",
    #         showlegend=False,
    #         hovermode='closest',
    #         margin=dict(b=20, l=5, r=5, t=40),
    #         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    #         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    #     )
    # )

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
    from phylogenetic_tree import g_to_newick, create_tree

    # Generate minimal example data (network)
    G = nx.random_geometric_graph(10, 0.5, seed=42)
    pos = nx.spring_layout(G)  # Generate positions for visualization
    nx.set_node_attributes(G, pos, 'pos')  # Assign positions as attributes
    G = nx.minimum_spanning_tree(G)

    G_nw = g_to_newick(G)
    fig = create_tree(G_nw)

    # Create a DataFrame with minimal node information
    df = pd.DataFrame({
        'node_id': list(G.nodes()),  # Node identifiers
        'x': [G.nodes[node]['pos'][0] for node in G.nodes()],  # x-coordinates
        'y': [G.nodes[node]['pos'][1] for node in G.nodes()],  # y-coordinates
        '# connections': [len(list(G.adj[node])) for node in G.nodes()]  # Number of connections
    })


    app = dash.Dash(__name__)
    run_dash_app(G, df, app, fig).run_server(debug=True)
