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
import logging
 
import dash
import plotly.graph_objects as go
import networkx as nx


def run_dash_app(app: dash.Dash, edge_trace, node_trace):
    """Run a Dash app to visualize the results of the dimensionality reduction.
    app.layout is setting my custom layout with plotly express.
    app.callback is setting the callback function to show the data of the selected points in the plot. 
    update_table is the callback function, filling the table with the data of the selected points.
    
    :param df: dataframe containing the annoattions
    :param X_red: dimensionality reduced embeddings
    :param method: dimensionality reduction method used
    :param project_name: name of the collection/dataset
    :param app: dash app
    return: dash app"""
    # Network graph
    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(
                        text="Minimal Spanning Tree with objective xy coloring",
                        font=dict(
                            size=16
                        )
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()

    return app


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
    mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title=dict(
                text='Node Connections',
                side='right'
                ),
                xanchor='left',
            ),
            line_width=2))
    
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    
    return edge_trace, node_trace


 
if __name__ == '__main__':
    # generate minimal example data (network)
    import networkx as nx
    G = nx.random_geometric_graph(200, 0.125, seed=42)
    G = nx.minimum_spanning_tree(G)

    edge_trace, node_trace = modify_graph_data(G)
    
    # Initialize Dash app
    app = dash.Dash(__name__)
    
    # Run the app
    run_dash_app(app, edge_trace, node_trace).run_server(debug=False)