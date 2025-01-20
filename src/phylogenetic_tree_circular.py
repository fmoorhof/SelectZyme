"""this is only a separate development script. i think code failed but if not, integrate into phylogenetic_tree.py

Conclusion Circos library is way too slow for already medium sized trees (backend calculation). impractical!
"""
from pycirclize import Circos
from pycirclize.utils import load_example_tree_file, ColorCycler
from matplotlib.lines import Line2D

import matplotlib


def circos_dendrogram(tree_file) -> matplotlib.figure.Figure:
    # Initialize Circos from phylogenetic tree
    circos, tv = Circos.initialize_from_tree(
        tree_file,
        r_lim=(30, 100),
        leaf_label_size=5,
        line_kws=dict(color="lightgrey", lw=1.0),
    )

    # Plot figure & set legend on center
    fig = circos.plotfig()
    _ = circos.ax.legend(
        fontsize=6,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
    )

    return fig

if __name__ == "__main__":
    tree_file = load_example_tree_file("large_example.nwk")
    fig = circos_dendrogram(tree_file)
    fig.savefig("datasets/pycirclize.png")