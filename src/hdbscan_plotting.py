import numpy as np
from scipy.cluster.hierarchy import dendrogram
import plotly.graph_objects as go


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

            hover_text = ', '.join([f"{col}: {self.df[col][i]}" for col in self.df.columns])

            fig.add_trace(go.Scatter(x=left_x, y=left_y, mode='lines',
                                     line=dict(color='black', width=np.log2(1 + lw[0])),
                                     text=hover_text, hoverinfo='text'))
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
            color_array = np.log2(np.array(linewidths).flatten())
            fig.update_layout(coloraxis=dict(colorscale=cmap, colorbar=dict(title='log(Number of points)')))

        fig.show()


def _get_dendrogram_ordering(parent, linkage, root):
    if parent < root:
        return []
    return _get_dendrogram_ordering(int(linkage[parent - root][0]), linkage, root) + \
           _get_dendrogram_ordering(int(linkage[parent - root][1]), linkage, root) + [parent]


def _calculate_linewidths(ordering, linkage, root):
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

    SingleLinkageTree(clusterer.single_linkage_tree_._linkage, df).plot()
