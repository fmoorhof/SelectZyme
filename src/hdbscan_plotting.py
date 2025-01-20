"""This implementation is inspired by the hdbscan plotting library to access the single linkage tree directly and not first converting it to a newick tree to later use it with a scatter plot (like done in phylogenetic_tree.py)
https://github.com/scikit-learn-contrib/hdbscan/blob/f0285287a62084e3a796f3a34901181972966b72/hdbscan/plots.py#L536
based on this implementation, the plotting function was modified to use plotly (instead of matplotlib). Also, df has been added for plot annotations.
Additionally, plotting unrelated functionalities (such as other export formats) were removed.
polar coordinates were inspired by this stackoverflow post:
https://stackoverflow.com/questions/51936574/how-to-plot-scipy-hierarchy-dendrogram-using-polar-coordinates

similarly, adaptations were integrated for the MST implementation:
https://github.com/scikit-learn-contrib/hdbscan/blob/f0285287a62084e3a796f3a34901181972966b72/hdbscan/plots.py#L760

MST networkx force directed layout implementation:
This is a minimal example dash app to visualize a networkx graph. Tutorial taken from: https://plotly.com/python/network-graphs/
possible outlook on networkx implementation:
- Insights into Connectivity and Routes: Shortest Paths, Betweenness Centrality, critical nodes

Conclusion on the adapted hdbscan implementation:
- Implementation by far the fastest, client side rendering (default) is very slow and loaded plot can not really be interacted with (batch7 dataset, min_samples=5; min_cluster_size=50) for MST, SLC renders slower but then works slowly. (batch7 dataset, min_samples=250; min_cluster_size=500). same slow results.
- MST in DimRed landscape is not really nice. The force directed layout is better. Apart from this only the connectivity information is really usefull there which can also maybe extracted differently.
- Trees: polar: nice that it worked now but go.scatterpolar is not really suited for the visualization
"""
from warnings import warn
import logging

import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import dendrogram
import plotly.graph_objects as go

from customizations import set_columns_of_interest

# try to revert in single_linkage_plotting.py to adress this issue: otherwise i dont see a way of fixing it:
    # todo: some (root) lines are not properly build and also the line scaling looks not so professional like before. Find reason and fix it.
class SingleLinkageTree(object):
    def __init__(self, linkage, df):
        self._linkage = linkage
        self.df = df

    def plot(self, truncate_mode=None, p=0, vary_line_width=True, cmap='Viridis', colorbar=True, polar=False):
        """Plot a dendrogram of the single linkage tree.

        Parameters
        ----------
        truncate_mode : str, optional
            Truncation mode ('none', 'lastp', or 'level') for the dendrogram.

        p : int, optional
            The `p` parameter for `truncate_mode`.

        vary_line_width : boolean, optional
            Whether to vary line width based on cluster size.

        cmap : str, optional
            Color scale to use for the clusters.

        colorbar : boolean, optional
            Whether to include a colorbar in the plot.

        polar : boolean, optional
            Whether to plot a circular (polar) dendrogram.
        """
        logging.info("Generating phylogenetic tree.")
        dendrogram_data = dendrogram(self._linkage, p=p, truncate_mode=truncate_mode, no_plot=True)
        X = np.array(dendrogram_data['icoord'])
        Y = np.array(dendrogram_data['dcoord'])
        leaf_indices = dendrogram_data['leaves']


        if polar and self.df.shape[0] > 5000:
            warn("Too many data points for rendering of a circular dendrogram figure. switched polar to False to create a non circular dendrogram.")
            polar=False

        if polar:
            # Transform for polar coordinates
            Y = -np.log(Y + 1)
            X_min, X_max = X.min(), X.max()
            X = ((X - X_min) / (X_max - X_min) * 0.8 + 0.1) * 2 * np.pi
            X = np.apply_along_axis(self._smooth_segment, 1, X)
            Y = np.apply_along_axis(self._smooth_segment, 1, Y)

        fig = go.Figure()

        # set hover data
        columns_of_interest = set_columns_of_interest(self.df.columns)  # Only show hover data for some df columns
        hover_texts = self.df.iloc[leaf_indices].apply(
            lambda row: '<br>'.join([f"{col}: {row[col]}" for col in columns_of_interest]), axis=1
        ).tolist()  # perf: code slow but still ok

    # todo: why customdata not working yet?
        # todo: assert why plotting results are different using this implementation! is algorithm non deterministic?
        # set hover data
        # columns_of_interest = set_columns_of_interest(self.df.columns)  # Only show hover data for some df columns
        # hover_texts=["<br>".join(f"{col}: {self.df[col][i]}" for col in columns_of_interest)
        #         for i in range(len(self.df))]

        if polar:  # todo: go.Scatterpolar not really appropriate for large interactive dendrogram visualizations
            # batch calculation for performance enhancement
            r_values = []
            theta_values = []
            text_values = []
            for i, (x, y) in enumerate(zip(X, Y)):
                r_values.extend(y)
                theta_values.extend(np.degrees(x))
                text_values.extend([hover_texts[i]] * len(y) if i < len(hover_texts) else [None] * len(y))

                # Add None to fix unrelated branch connecting lines
                r_values.append(None)
                theta_values.append(None)
                text_values.append(None)

            # create figure with pre-computed values
            fig.add_trace(go.Scatterpolargl(
                r=r_values,
                theta=theta_values,
                mode='lines',
                line=dict(color='black', width=1.0),
                # todo: why customdata not working yet?
                # customdata=self.df['accession'],  # needed to pass accession to callback from which entire row is restored of df
                text=text_values,
                hoverinfo='text'
            ))  # perf: quite slow: assert why and enhance!
        else:
            # batch calculation for performance enhancement
            x_values = []
            y_values = []
            text_values = []
            for i, (x, y) in enumerate(zip(X, Y)):
                x_values.extend(x)
                y_values.extend(y)
                text_values.extend([hover_texts[i]] * len(y) if i < len(hover_texts) else [None] * len(y))

                # Add None to fix unrelated branch connecting lines
                x_values.append(None)
                y_values.append(None)
                text_values.append(None)

            # create figure with pre-computed values
            fig.add_trace(go.Scattergl(
                x=x_values,
                y=y_values,
                mode='lines',
                line=dict(color='black', width=1.0),
                # todo: why customdata not working yet?
                # customdata=self.df['accession'],  # needed to pass accession to callback from which entire row is restored of df
                text=text_values,
                hoverinfo='text'
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=polar),
                angularaxis=dict(visible=polar)
            ) if polar else {},
            xaxis=dict(title='Clusters', showticklabels=not polar),
            yaxis=dict(title='Distance' if not polar else 'Log(Distance)', visible=not polar),
            showlegend=False,
            dragmode='zoom'
        )

        if colorbar:
            fig.update_layout(coloraxis=dict(colorscale=cmap))

        return fig

    @staticmethod
    def _smooth_segment(segment, Nsmooth=20):
        """Smooth a line segment for polar plotting."""
        return np.concatenate([[segment[0]], np.linspace(segment[1], segment[2], Nsmooth), [segment[3]]])


class MinimumSpanningTree:
    def __init__(self, mst, data, X_red, df):
        self._mst = mst
        self._data = data
        self.X_red = X_red
        self.df = df
    
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
        edge_trace, node_trace = self._modify_graph_data(G)

        fig = go.Figure()
        fig.add_trace(edge_trace)
        fig.add_trace(node_trace)

        fig.update_layout(
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
        
        return fig

    # Not really nice visualization and connectivity information not displayed because no access of information implemented
    def plot_mst_in_DimRed_landscape(self, edge_alpha=0.3, edge_linewidth=1, vary_line_width=True):
        """
        Plot the minimum spanning tree in the dimensionality-reduced landscape.
        """
        logging.info("Generating MST in dimensionality reduced plot.")
        if self._data.shape[0] > 50000:
            warn("Too many data points for safe rendering of a minimum spanning tree!")
            return None

        # Vary line width if enabled
        if vary_line_width:
            line_width = edge_linewidth * (np.log(self._mst.T[2].max() / self._mst.T[2]) + 1.0)
        else:
            line_width = edge_linewidth

        # Edge coordinates and weights
        line_coords = self.X_red[self._mst[:, :2].astype(int)]
        edge_x, edge_y = [], []
        for (x1, y1), (x2, y2) in zip(line_coords[:, 0], line_coords[:, 1]):
            edge_x.extend([x1, x2, None])
            edge_y.extend([y1, y2, None])

        # Create edge and node traces
        edge_trace = self.create_edge_trace(edge_x, edge_y, edge_alpha, line_width.mean())
        node_trace = self.create_node_trace(self.X_red[:, 0], self.X_red[:, 1])

        # Create the figure
        fig = go.Figure()
        fig.add_trace(edge_trace)
        fig.add_trace(node_trace)

        # Update layout
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False
        )

        return fig

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

        if len(G.nodes()) > 5000:
            logging.warning("Minimal Spanning Tree (MST) will with over 5000 nodes will be too large for nice visualizations. Concludingly, a reduced MST is created that only shows nodes with connectivity greater than 1.")
            G = self._prune_graph(G)

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

        edge_trace = self.create_edge_trace(edge_x, edge_y)

        # Node traces
        node_x = [G.nodes[node]['pos'][0] for node in G.nodes()]
        node_y = [G.nodes[node]['pos'][1] for node in G.nodes()]
        node_trace = self.create_node_trace(node_x, node_y)

        # Color nodes by their number of connections
        node_adjacencies = [len(list(G.adj[node])) for node in G.nodes()]
        node_trace.marker.color = node_adjacencies

        return edge_trace, node_trace

    def _prune_graph(self, G: nx.Graph):
        """
        Prune the graph to keep only nodes with connectivity > 1 and update the DataFrame accordingly.
        !Assumes that the DataFrame index corresponds to the node IDs in the graph.!

        Parameters:
        G (networkx.Graph): The input graph.
        Returns:
        tuple: A tuple containing:
            - G (networkx.Graph): The pruned graph.
        """
        # Filter nodes with connectivity > 1
        nodes_to_keep = [node for node in G.nodes() if len(list(G.adj[node])) > 4]
        G = G.subgraph(nodes_to_keep).copy()

        # Update the DataFrame to keep only rows corresponding to the retained nodes
        self.df = self.df.loc[nodes_to_keep].copy()
        self.df.reset_index(inplace=True)

        return G
    
    @staticmethod
    def create_edge_trace(edge_x, edge_y, edge_alpha=0.3, edge_width=1.0):
        """
        Create a Plotly edge trace for the graph.
        """
        return go.Scattergl(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color=f"rgba(0,0,0,{edge_alpha})", width=edge_width),
            hoverinfo="none"
        )

    def create_node_trace(self, node_x, node_y):
        """
        Create a Plotly node trace for the graph.
        """
        columns_of_interest = set_columns_of_interest(self.df.columns)
        hover_text = ["<br>".join(f"{col}: {self.df[col][i]}" for col in columns_of_interest) for i in range(len(self.df))]
        
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
                reversescale=True,
                color=[],  # Will be populated with node adjacencies
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                )
            )
        )    


if __name__ == "__main__":
    import hdbscan
    import numpy as np
    from sklearn.datasets import make_blobs
    import pandas as pd

    np.random.seed(42)
    sample_size = 50  # too big samples cause RecursionError but strangely not for my real datasets
    df = pd.DataFrame({
        'x': np.random.randn(sample_size),
        'y': np.random.randn(sample_size),
        'accession': np.random.choice(['A', 'B', 'C'], sample_size),
        'cluster': np.random.choice(['A', 'B', 'C'], sample_size),
        'selected': np.random.choice([False, True], sample_size),
        'marker_symbol': np.random.choice(['circle', 'square', 'diamond', 'triangle-up'], sample_size),
        'marker_size': np.random.randint(10, sample_size, sample_size)
    })
    data, _ = make_blobs(n_samples=sample_size, n_features=2, centers=3, cluster_std=0.8, random_state=42)
    X_red = np.random.randn(sample_size, 2)  # 2D mock PCA data

    hdbscan = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    hdbscan.fit(data)

    # debug and develop here
    # fig = SingleLinkageTree(hdbscan.single_linkage_tree_._linkage, df).plot(polar=False)
    mst = MinimumSpanningTree(hdbscan.minimum_spanning_tree_._mst, hdbscan.minimum_spanning_tree_._data, X_red, df)
    fig = mst.plot_mst_in_DimRed_landscape()  # NOT favored visualization
    fig = mst.plot_mst_force_directed(hdbscan.minimum_spanning_tree_)

    fig.show()
