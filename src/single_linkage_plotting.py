"""This implementation is inspired by the hdbscan plotting library to access the single linkage tree directly and not first converting it to a newick tree to later use it with a scatter plot (like done in phylogenetic_tree.py)
Also, this implementation is a revert from initial hdbscan_plotting.py commits (https://github.com/fmoorhof/ec/blob/4103f530d07031fcb66a879eb95713eea17178fe/src/hdbscan_plotting.py) 4103f53, that will get merged consecutively with functionalities from hdbcan_plotting.py.

https://github.com/scikit-learn-contrib/hdbscan/blob/f0285287a62084e3a796f3a34901181972966b72/hdbscan/plots.py#L536
based on this implementation, the plotting function was modified to use plotly (instead of matplotlib). Also, df has been added for plot annotations.
Additionally, plotting unrelated functionalities (such as other export formats) were removed.
"""
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import plotly.graph_objects as go

from customizations import set_columns_of_interest

class SingleLinkageTree(object):
    def __init__(self, linkage, df):
        self._linkage = linkage
        self.df = df

    def plot(self, truncate_mode=None, p=0, vary_line_width=True, cmap='Viridis', colorbar=True):
        
        dendrogram_data = dendrogram(self._linkage, p=p, truncate_mode=truncate_mode, no_plot=True)
        X = dendrogram_data['icoord']
        Y = dendrogram_data['dcoord']

        if vary_line_width:
            dendrogram_ordering = _get_dendrogram_ordering(2 * len(self._linkage), self._linkage, len(self._linkage) + 1)
            linewidths = _calculate_linewidths(dendrogram_ordering, self._linkage, len(self._linkage) + 1)
        else:
            linewidths = [(1.0, 1.0)] * len(Y)

        fig = go.Figure()
        
        for i, (x, y, lw) in enumerate(zip(X, Y, linewidths)):
            left_x = x[:2]
            right_x = x[2:]
            left_y = y[:2]
            right_y = y[2:]
            horizontal_x = x[1:3]
            horizontal_y = y[1:3]

            columns_of_interest = set_columns_of_interest(self.df.columns)
            hover_text = '<br>'.join([f"{col}: {self.df[col][i]}" for col in columns_of_interest])

            fig.add_trace(go.Scattergl(x=left_x, y=left_y, mode='lines',
                                     line=dict(color='black', width=np.log2(1 + lw[0])),
                                     customdata=self.df['accession'],
                                     text=hover_text, hoverinfo='text'))
            fig.add_trace(go.Scattergl(x=right_x, y=right_y, mode='lines',
                                     line=dict(color='black', width=np.log2(1 + lw[1])),
                                     customdata=self.df['accession'],
                                     text=hover_text, hoverinfo='text'))
            fig.add_trace(go.Scattergl(x=horizontal_x, y=horizontal_y, mode='lines',
                                     line=dict(color='black', width=1.0),
                                     customdata=self.df['accession'],
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


if __name__ == "__main__":
    import hdbscan
    import numpy as np
    from sklearn.datasets import make_blobs
    import pandas as pd

    np.random.seed(42)
    sample_size = 11  # too big samples cause RecursionError but strangely not for my real datasets
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
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)
    clusterer.fit(data)

    fig = SingleLinkageTree(clusterer.single_linkage_tree_._linkage, df).plot()
    fig.show()