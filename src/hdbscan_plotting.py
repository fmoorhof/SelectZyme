"""This implementation is taken from hdbscan to replace matplotlib plotting by plotly to enable interactive plots.
https://github.com/scikit-learn-contrib/hdbscan/blob/f0285287a62084e3a796f3a34901181972966b72/hdbscan/plots.py#L760"""

import numpy as np
from scipy.cluster.hierarchy import dendrogram


class SingleLinkageTree(object):
    """A single linkage format dendrogram tree, with plotting functionality
    and networkX support.

    Parameters
    ----------
    linkage : ndarray (n_samples, 4)
        The numpy array that holds the tree structure. As output by
        scipy.cluster.hierarchy, hdbscan, of fastcluster.

    """
    def __init__(self, linkage):
        self._linkage = linkage

    def plot(self, axis=None, truncate_mode=None, p=0, vary_line_width=True,
             cmap='viridis', colorbar=True):
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

        Returns
        -------
        axis : matplotlib axis
               The axis on which the dendrogram plot has been rendered.

        """
        dendrogram_data = dendrogram(self._linkage, p=p, truncate_mode=truncate_mode, no_plot=True)
        X = dendrogram_data['icoord']
        Y = dendrogram_data['dcoord']

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('You must install the matplotlib library to plot the single linkage tree.')

        if axis is None:
            axis = plt.gca()

        if vary_line_width:
            dendrogram_ordering = _get_dendrogram_ordering(2 * len(self._linkage), self._linkage, len(self._linkage) + 1)
            linewidths = _calculate_linewidths(dendrogram_ordering, self._linkage, len(self._linkage) + 1)
        else:
            linewidths = [(1.0, 1.0)] * len(Y)

        if cmap != 'none':
            color_array = np.log2(np.array(linewidths).flatten())
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                       norm=plt.Normalize(0, color_array.max()))
            sm.set_array(color_array)

        for x, y, lw in zip(X, Y, linewidths):
            left_x = x[:2]
            right_x = x[2:]
            left_y = y[:2]
            right_y = y[2:]
            horizontal_x = x[1:3]
            horizontal_y = y[1:3]

            if cmap != 'none':
                axis.plot(left_x, left_y, color=sm.to_rgba(np.log2(lw[0])),
                          linewidth=np.log2(1 + lw[0]),
                          solid_joinstyle='miter', solid_capstyle='butt')
                axis.plot(right_x, right_y, color=sm.to_rgba(np.log2(lw[1])),
                          linewidth=np.log2(1 + lw[1]),
                          solid_joinstyle='miter', solid_capstyle='butt')
            else:
                axis.plot(left_x, left_y, color='k',
                          linewidth=np.log2(1 + lw[0]),
                          solid_joinstyle='miter', solid_capstyle='butt')
                axis.plot(right_x, right_y, color='k',
                          linewidth=np.log2(1 + lw[1]),
                          solid_joinstyle='miter', solid_capstyle='butt')

            axis.plot(horizontal_x, horizontal_y, color='k', linewidth=1.0,
                      solid_joinstyle='miter', solid_capstyle='butt')

        if colorbar:
            cb = plt.colorbar(sm, ax=axis)
            cb.ax.set_ylabel('log(Number of points)')

        axis.set_xticks([])
        for side in ('right', 'top', 'bottom'):
            axis.spines[side].set_visible(False)
        axis.set_ylabel('distance')

        return axis


def _get_dendrogram_ordering(parent, linkage, root):

    if parent < root:
        return []

    return _get_dendrogram_ordering(int(linkage[parent-root][0]), linkage, root) + \
            _get_dendrogram_ordering(int(linkage[parent-root][1]), linkage, root) + [parent]

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
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.datasets import make_blobs

    plot_kwds = {'alpha': 0.5, 's': 80, 'linewidths': 0}

    data, _ = make_blobs(n_samples=50, n_features=2, centers=3, cluster_std=0.8)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    clusterer.fit(data)

    plt.figure()
    SingleLinkageTree(clusterer.single_linkage_tree_._linkage).plot()

    plt.savefig("datasets/hdbscan_slc.png")