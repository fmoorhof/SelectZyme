"""This implementation is inspired by the hdbscan plotting library to access the single linkage tree directly and not first converting it to a newick tree to later use it with a scatter plot (like done in phylogenetic_tree.py)
similarly, adaptations were integrated for the MST implementation:
https://github.com/scikit-learn-contrib/hdbscan/blob/f0285287a62084e3a796f3a34901181972966b72/hdbscan/plots.py#L760

MST networkx force directed layout implementation:
This is a minimal example dash app to visualize a networkx graph. Tutorial taken from: https://plotly.com/python/network-graphs/
possible outlook on networkx implementation:
- Insights into Connectivity and Routes: Shortest Paths, Betweenness Centrality, critical nodes

Conclusion on the adapted hdbscan implementation:
- MST in DimRed landscape is not really nice. The force directed layout is better. Apart from this only the connectivity information is really usefull there which can also maybe extracted differently.
"""
import logging

import numpy as np
import networkx as nx
import plotly.graph_objects as go

from src.customizations import set_columns_of_interest
from src.utils import run_time


class MinimumSpanningTree:
    def __init__(self, mst, df, X_red, fig):
        self._mst = mst  # struct: (node1, node2, weight)
        self.df = df
        self.X_red = X_red
        self.fig = fig
    
    @run_time
    def plot_mst_force_directed(self, G: nx.Graph):
        """
        Plots a Minimum Spanning Tree (MST) using a force-directed layout.
        This function takes a NetworkX graph object representing the MST and 
        visualizes it using Plotly. The graph is modified and plotted with 
        custom layout settings to enhance visualization.
        Parameters:
        -----------
        G : nx.Graph
            A NetworkX graph object representing the Minimum Spanning Tree (MST).
        Returns:
        --------
        fig : plotly.graph_objs._figure.Figure
            A Plotly Figure object containing the MST visualization.
        """
        logging.info("Generating force-directed layout MST.")
        edge_trace, node_trace = self._modify_graph_data(G)  # perf: slow

        fig = go.Figure()
        fig.add_trace(edge_trace)
        fig.add_trace(node_trace)

        fig.update_layout(
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False,  # hide the node and edge trace legend plotted over the custom legend
            )
        
        return fig

    @run_time
    def plot_mst_in_DimRed_landscape(self):
        """
        Plot the minimum spanning tree in the dimensionality-reduced landscape.
        """
        # Edge coordinates
        line_coords = self.X_red[self._mst[:, :2].astype(int)]  # X_red[node connections] = coordinates of the nodes in X_red
        edge_x, edge_y = [], []
        for (x1, y1), (x2, y2) in zip(line_coords[:, 0], line_coords[:, 1]):
            edge_x.extend([x1, x2, None])  # None: A separator indicating the end of this edge
            edge_y.extend([y1, y2, None])

        # Create edge traces
        edge_trace = self.create_edge_trace(edge_x, edge_y, edge_opacity=0.5, edge_width=0.3)

        return self.fig.add_trace(edge_trace)  # if nodes are not first, hover data randomly get only displayed for some nodes!
    
    @run_time
    def create_edge_trace(self, edge_x: list, edge_y: list, edge_opacity=None, edge_width: int = 1):
        """
        Create a Plotly edge trace for the graph.
        """      
        return go.Scattergl(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="black", width=edge_width),
            opacity=edge_opacity,
        )

    @run_time
    def create_node_trace(self, node_x: np.ndarray, node_y: np.ndarray, node_adjacencies: np.ndarray):
        """
        Create a Plotly node trace for the graph.
        """
        columns_of_interest = set_columns_of_interest(self.df.columns)
        hover_text = [
            "<br>".join(f"{col}: {self.df[col][i]}" for col in columns_of_interest) + f"<br>connectivity: {node_adjacencies[i]}"
            for i in range(len(self.df))
        ]
        
        return go.Scattergl(
            x=node_x,
            y=node_y,
            mode="markers",
            customdata=self.df['accession'],
            hovertext=hover_text,
            hoverinfo="text",
            marker=dict(
                size=self.df['marker_size'],
                symbol=self.df['marker_symbol'],
                opacity=0.7,
                line_width=1,

                # connectivity legend
                showscale=True,
                colorscale='YlGnBu',
                reversescale=False,
                color=[],  # Will be populated with node adjacencies
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                )
            )
        )
    
    def _modify_graph_data(self, G) -> tuple:
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
        # pos = nx.spring_layout(G)  # some overlaying nodes, NOT favored layout
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

        # Calculate nodes connectivity
        node_adjacencies = [len(list(G.adj[node])) for node in G.nodes()]

        edge_trace = self.create_edge_trace(edge_x, edge_y, node_adjacencies)

        # Node traces
        node_x = [G.nodes[node]['pos'][0] for node in G.nodes()]
        node_y = [G.nodes[node]['pos'][1] for node in G.nodes()]
        node_trace = self.create_node_trace(node_x, node_y, node_adjacencies)

        # Color nodes by their number of connections
        node_trace.marker.color = node_adjacencies

        return edge_trace, node_trace



if __name__ == "__main__":
    import hdbscan
    import numpy as np
    from sklearn.datasets import make_blobs
    import pandas as pd

    np.random.seed(42)
    sample_size = 5  # too big samples cause RecursionError but strangely not for my real datasets
    df = pd.DataFrame({
        'x': np.random.randn(sample_size),
        'y': np.random.randn(sample_size),
        'accession': np.random.choice(['A', 'B', 'C'], sample_size),
        'cluster': np.random.choice(['A', 'B', 'C'], sample_size),
        'selected': np.random.choice([False, True], sample_size),
        'marker_symbol': np.random.choice(['circle', 'square', 'diamond', 'triangle-up'], sample_size),
        'marker_size': np.random.randint(2, sample_size, sample_size)
    })
    data, _ = make_blobs(n_samples=sample_size, n_features=2, centers=3, cluster_std=0.8, random_state=42)
    X_red = np.random.randn(sample_size, 2)  # 2D mock PCA data

    hdbscan = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    hdbscan.fit(data)

    # debug and develop here
    mst = MinimumSpanningTree(hdbscan.minimum_spanning_tree_._mst, X_red, df)
    fig = mst.plot_mst_in_DimRed_landscape()
    # fig = mst.plot_mst_force_directed(hdbscan.minimum_spanning_tree_)
    fig.show()
