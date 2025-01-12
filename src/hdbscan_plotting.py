"""This implementation is inspired by the hdbscan plotting library to access the single linkage tree directly and not first converting it to a newick tree to later use it with a scatter plot (like done in phylogenetic_tree.py)
https://github.com/scikit-learn-contrib/hdbscan/blob/f0285287a62084e3a796f3a34901181972966b72/hdbscan/plots.py#L536
based on this implementation, the plotting function was modified to use plotly (instead of matplotlib). Also, df has been added for plot annotations.
Additionally, plotting unrelated functionalities (such as other export formats) were removed.
polar coordinates were inspired by this stackoverflow post:
https://stackoverflow.com/questions/51936574/how-to-plot-scipy-hierarchy-dendrogram-using-polar-coordinates

similarly, adaptations were integrated for the MST implementation:
https://github.com/scikit-learn-contrib/hdbscan/blob/f0285287a62084e3a796f3a34901181972966b72/hdbscan/plots.py#L760
"""
from warnings import warn

import numpy as np
from scipy.cluster.hierarchy import dendrogram
import plotly.graph_objects as go


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
        dendrogram_data = dendrogram(self._linkage, p=p, truncate_mode=truncate_mode, no_plot=True)
        X = np.array(dendrogram_data['icoord'])
        Y = np.array(dendrogram_data['dcoord'])
        leaf_indices = dendrogram_data['leaves']

        if polar:
            # Transform for polar coordinates
            Y = -np.log(Y + 1)
            X_min, X_max = X.min(), X.max()
            X = ((X - X_min) / (X_max - X_min) * 0.8 + 0.1) * 2 * np.pi
            X = np.apply_along_axis(self._smooth_segment, 1, X)
            Y = np.apply_along_axis(self._smooth_segment, 1, Y)

        fig = go.Figure()

        hover_texts = self.df.iloc[leaf_indices].apply(
            lambda row: '<br>'.join([f"{col}: {row[col]}" for col in self.df.columns]), axis=1
        ).tolist()

        for i, (x, y) in enumerate(zip(X, Y)):
            if polar:
                fig.add_trace(go.Scatterpolar(
                    r=y,
                    theta=np.degrees(x),
                    mode='lines',
                    line=dict(color='black', width=1.0),
                    text=hover_texts[i] if i < len(hover_texts) else None,
                    hoverinfo='text'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    line=dict(color='black', width=1.0),
                    text=hover_texts[i] if i < len(hover_texts) else None,
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
    def _smooth_segment(segment, Nsmooth=100):
        """Smooth a line segment for polar plotting."""
        return np.concatenate([[segment[0]], np.linspace(segment[1], segment[2], Nsmooth), [segment[3]]])


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

        hover_text=[
                "<br>".join(f"{col}: {self.df[col][i]}" for col in self.df.columns)
                for i in range(len(self.df))
            ],

        fig.add_trace(go.Scatter(
            x=self.X_red[:, 0],
            y=self.X_red[:, 1],
            mode="markers",
            marker=dict(
                size=node_size,
                color=node_color,
                opacity=node_alpha
            ),
            hovertext=hover_text,
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

    np.random.seed(42)
    sample_size = 500000
    df = pd.DataFrame({
        'x': np.random.randn(sample_size),
        'y': np.random.randn(sample_size),
        'cluster': np.random.choice(['A', 'B', 'C'], sample_size),
        'marker_symbol': np.random.choice(['circle', 'square', 'diamond', 'triangle-up'], sample_size),
        'marker_size': np.random.randint(10, sample_size, sample_size)
    })
    data, _ = make_blobs(n_samples=sample_size, n_features=2, centers=3, cluster_std=0.8, random_state=42)
    X_red = np.random.randn(sample_size, 2)  # 2D mock PCA data

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    clusterer.fit(data)

    # debug and develop here
    fig = SingleLinkageTree(clusterer.single_linkage_tree_._linkage, df).plot(polar=False)
    # fig = MinimumSpanningTree(clusterer.minimum_spanning_tree_._mst, clusterer.minimum_spanning_tree_._data, X_red, df).plot()

    fig.show()
