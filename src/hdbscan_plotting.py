"""This implementation is inspired by the hdbscan plotting library to access the single linkage tree directly and not first converting it to a newick tree to later use it with a scatter plot (like done in phylogenetic_tree.py)
https://github.com/scikit-learn-contrib/hdbscan/blob/f0285287a62084e3a796f3a34901181972966b72/hdbscan/plots.py#L536
based on this implementation, the plotting function was modified to use plotly (instead of matplotlib). Also, df has been added for plot annotations.
Additionally, plotting unrelated functionalities (such as other export formats) were removed.

similarly, adaptations were integrated for the MST implementation:
https://github.com/scikit-learn-contrib/hdbscan/blob/f0285287a62084e3a796f3a34901181972966b72/hdbscan/plots.py#L760
"""
from warnings import warn

import numpy as np
from scipy.cluster.hierarchy import dendrogram
import plotly.graph_objects as go
import plotly.express as px


class SingleLinkageTree(object):
    def __init__(self, linkage, df):
        self._linkage = linkage
        self.df = df

    def plot(self, truncate_mode=None, p=0, vary_line_width=True, cmap='Viridis', colorbar=True):
        """Plot a dendrogram of the single linkage tree.

        Parameters
        ----------
        truncate_mode : str, optional
                        The dendrogram can be hard to read when the original
                        observation matrix from which the linkage is derived
                        is large. Truncation is used to condense the dendrogram.
                        There are several modes:

        ``None/'none'``
                No truncation is performed (Default).

        ``'lastp'``
                The last p non-singleton formed in the linkage are the only
                non-leaf nodes in the linkage; they correspond to rows
                Z[n-p-2:end] in Z. All other non-singleton clusters are
                contracted into leaf nodes.

        ``'level'/'mtica'``
                No more than p levels of the dendrogram tree are displayed.
                This corresponds to Mathematica(TM) behavior.

        p : int, optional
            The ``p`` parameter for ``truncate_mode``.

        vary_line_width : boolean, optional
            Draw downward branches of the dendrogram with line thickness that
            varies depending on the size of the cluster.

        cmap : string or matplotlib colormap, optional
               The matplotlib colormap to use to color the cluster bars.
               A value of 'none' will result in black bars.
               (default 'viridis')

        colorbar : boolean, optional
                   Whether to draw a matplotlib colorbar displaying the range
                   of cluster sizes as per the colormap. (default True)
        """
        dendrogram_data = dendrogram(self._linkage, p=p, truncate_mode=truncate_mode, no_plot=True)
        X = dendrogram_data['icoord']
        Y = dendrogram_data['dcoord']

        if vary_line_width:
            dendrogram_ordering = _get_dendrogram_ordering(2 * len(self._linkage), self._linkage, len(self._linkage) + 1)
            linewidths = _calculate_linewidths(dendrogram_ordering, self._linkage, len(self._linkage) + 1)
        else:
            linewidths = [(1.0, 1.0)] * len(Y)

        fig = go.Figure()
        # plotly.express implementation, in case if needed for selection events, works seemlessly
        # cols = self.df.columns.values.tolist()
        # fig = px.scatter(self.df, hover_data=cols)
        
        for i, (x, y, lw) in enumerate(zip(X, Y, linewidths)):
            left_x = x[:2]
            right_x = x[2:]
            left_y = y[:2]
            right_y = y[2:]
            horizontal_x = x[1:3]
            horizontal_y = y[1:3]

            hover_text = '<br>'.join([f"{col}: {self.df[col][i]}" for col in self.df.columns])

            fig.add_trace(go.Scatter(x=left_x, y=left_y, mode='lines',
                                     line=dict(color='black', width=np.log2(1 + lw[0])),
                                     text=hover_text, hoverinfo='text'))  # hoverlabel_align='left'  # only working if multiline is too long=useless
            fig.add_trace(go.Scatter(x=right_x, y=right_y, mode='lines',
                                     line=dict(color='black', width=np.log2(1 + lw[1])),
                                     text=hover_text, hoverinfo='text'))
            fig.add_trace(go.Scatter(x=horizontal_x, y=horizontal_y, mode='lines',
                                     line=dict(color='black', width=1.0),
                                     text=hover_text, hoverinfo='text'))

        fig.update_layout(
            xaxis=dict(showticklabels=False),
            yaxis=dict(title='distance'),
            showlegend=False
        )

        if colorbar:
            fig.update_layout(coloraxis=dict(colorscale=cmap, colorbar=dict(title='log(Number of points)')))

        return fig


def _get_dendrogram_ordering(parent, linkage, root):
    """
    Recursively computes the dendrogram ordering for hierarchical clustering.
    Args:
        parent (int): The current parent node index.
        linkage (ndarray): The linkage matrix containing hierarchical clustering information.
        root (int): The root node index.
    Returns:
        list: A list of node indices representing the dendrogram ordering.
    """

    if parent < root:
        return []
    return _get_dendrogram_ordering(int(linkage[parent - root][0]), linkage, root) + \
           _get_dendrogram_ordering(int(linkage[parent - root][1]), linkage, root) + [parent]


def _calculate_linewidths(ordering, linkage, root):
    """
    Calculate the linewidths for each node in the hierarchical clustering.
    This function computes the linewidths for the left and right branches of each node
    in the hierarchical clustering dendrogram based on the linkage matrix.
    Args:
        ordering (list): A list of node indices in the order they should be processed.
        linkage (ndarray): The linkage matrix containing the hierarchical clustering.
                           Each row corresponds to a merge, with the format [idx1, idx2, dist, sample_count].
        root (int): The root node index from which to start the calculation.
    Returns:
        list: A list of tuples, where each tuple contains the linewidths for the left and right branches
              of the corresponding node in the ordering.
    """

    linewidths = []
    for x in ordering:
        if linkage[x - root][0] >= root:
            left_width = linkage[int(linkage[x - root][0]) - root][3]
        else:
            left_width = 1
        if linkage[x - root][1] >= root:
            right_width = linkage[int(linkage[x - root][1]) - root][3]
        else:
            right_width = 1
        linewidths.append((left_width, right_width))
    return linewidths


class MinimumSpanningTree:
    def __init__(self, mst, data, X_red, df):
        self._mst = mst
        self._data = data
        self.X_red = X_red
        self.df = df

    def plot(self, node_size=4, node_color="black", node_alpha=0.5,
             edge_alpha=0.3, edge_linewidth=1, vary_line_width=True, colorbar=True):
        """
        Plot the minimum spanning tree using Plotly Express.

        :param node_size: Size of the nodes in the scatter plot.
        :param node_color: Color of the nodes.
        :param node_alpha: Opacity of the nodes.
        :param edge_alpha: Opacity of the edges.
        :param edge_linewidth: Base linewidth of edges.
        :param vary_line_width: If True, vary edge widths by weights.
        :param colorbar: If True, add a colorbar for edge weights.
        :return: Plotly figure object.
        """
        if self._data.shape[0] > 32767:
            warn("Too many data points for safe rendering of a minimum spanning tree!")
            return None

        # Vary line width if enabled
        if vary_line_width:
            line_width = edge_linewidth * (np.log(self._mst.T[2].max() / self._mst.T[2]) + 1.0)
        else:
            line_width = edge_linewidth

        # Edge coordinates and weights
        line_coords = self.X_red[self._mst[:, :2].astype(int)]
        edge_x, edge_y, edge_weights = [], [], []
        for (x1, y1), (x2, y2), weight, lw in zip(line_coords[:, 0], line_coords[:, 1], self._mst[:, 2], line_width):
            edge_x.extend([x1, x2, None])
            edge_y.extend([y1, y2, None])
            edge_weights.append(weight)

        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="rgba(0,0,0,{})".format(edge_alpha), width=line_width.mean()),
            hoverinfo="none"
        ))

        fig.add_trace(go.Scatter(
            x=self.X_red[:, 0],
            y=self.X_red[:, 1],
            mode="markers",
            marker=dict(
                size=node_size,
                color=node_color,
                opacity=node_alpha
            ),
            hovertext=[
                "<br>".join(f"{col}: {self.df[col][i]}" for col in self.df.columns)
                for i in range(len(self.df))
            ],
            hoverinfo="text"
        ))

        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False
        )

        return fig



if __name__ == "__main__":
    import hdbscan
    import numpy as np
    from sklearn.datasets import make_blobs
    import pandas as pd

    df = pd.DataFrame({
        'x': np.random.randn(50),
        'y': np.random.randn(50),
        'cluster': np.random.choice(['A', 'B', 'C'], 50),
        'marker_symbol': np.random.choice(['circle', 'square', 'diamond', 'triangle-up'], 50),
        'marker_size': np.random.randint(10, 50, 50)
    })
    data, _ = make_blobs(n_samples=50, n_features=2, centers=3, cluster_std=0.8, random_state=42)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    clusterer.fit(data)

    # debug and develop here
    fig = SingleLinkageTree(clusterer.single_linkage_tree_._linkage, df).plot()
    fig = MinimumSpanningTree(clusterer.minimum_spanning_tree_._mst, clusterer.minimum_spanning_tree_._data, df).plot()
    fig.show()
