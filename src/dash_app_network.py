"""This is a minimal example dash app to visualize a networkx graph. Node connections are displayed and the property as hover over event.
Tutorial taken from: https://plotly.com/python/network-graphs/

2. Insights into Connectivity and Routes

    Shortest Paths, Betweenness Centrality, critical nodes
"""
import plotly.graph_objects as go
import networkx as nx
import pandas as pd


def modify_graph_data(G, df):
    """
    Modify the graph data for visualization.
    Parameters:
    G (networkx.Graph or compatible): The input graph. If not a networkx.Graph, it will be converted.
    df (pandas.DataFrame): DataFrame containing node attributes and additional information.
    Returns:
    tuple: A tuple containing:
        - edge_trace (plotly.graph_objs.Scatter): Scatter plot trace for edges.
        - node_trace (plotly.graph_objs.Scatter): Scatter plot trace for nodes with attributes.
    """
    if not isinstance(G, nx.Graph):
        G = G.to_networkx()

    # define graph layout and coordinates
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



if __name__ == '__main__':
    # Generate minimal example data (network)
    G = nx.random_geometric_graph(10, 0.5, seed=42)
    pos = nx.spring_layout(G)  # Generate positions for visualization
    nx.set_node_attributes(G, pos, 'pos')  # Assign positions as attributes
    G = nx.minimum_spanning_tree(G)

    # Create a DataFrame with minimal node information
    df = pd.DataFrame({
        'accession': [f'node_{i}' for i in range(len(G.nodes()))],
        'node_id': list(G.nodes()),
        'x': [G.nodes[node]['pos'][0] for node in G.nodes()],  # x-coordinates
        'y': [G.nodes[node]['pos'][1] for node in G.nodes()],  # y-coordinates
        '# connections': [len(list(G.adj[node])) for node in G.nodes()],  # Number of connections
        'marker_size': [10 for node in G.nodes()],
        'marker_symbol': ['circle' for node in G.nodes()],
    })

    edge_trace, node_trace = modify_graph_data(G, df)
    fig = go.Figure()
    fig.add_trace(edge_trace)
    fig.add_trace(node_trace)

    fig.update_layout(
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
    fig.show()