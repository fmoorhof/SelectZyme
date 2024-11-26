"""Implementations inspired and mostly taken from: https://github.com/plotly/dash-sample-apps/blob/main/apps/dash-phylogeny/utils.py
networkx graph to newick format implementation is taken from: https://stackoverflow.com/questions/46444454/save-networkx-tree-in-newick-format"""
import io

from Bio import Phylo
import plotly.graph_objs as go


def g_to_newick(g, root=None):
    """
    Convert a directed graph to Newick format.

    Parameters:
    g (networkx.DiGraph): A directed graph representing the tree.
    root (node, optional): The root node of the tree. If None, the root is determined automatically.

    Returns:
    str: A string representing the tree in Newick format.

    Raises:
    AssertionError: If the graph does not have exactly one root node.
    """
    if root is None:
        roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
        assert 1 == len(roots)
        root = roots[0][0]
    subgs = []
    for child in g[root]:
        if len(g[child]) > 0:
            subgs.append(g_to_newick(g, root=child))
        else:
            subgs.append(child)
    return "(" + ','.join(map(str, subgs)) + ")"


def create_tree(nw_tree):
    tree = Phylo.read(io.StringIO(nw_tree), "newick")
    x_coords = get_x_coordinates(tree)
    y_coords = get_y_coordinates(tree)
    line_shapes = []
    draw_clade(
        tree.root,
        0,
        line_shapes,
        line_color="rgb(25,25,25)",
        line_width=1,
        x_coords=x_coords,
        y_coords=y_coords,
    )
    my_tree_clades = x_coords.keys()
    X = []
    Y = []
    text = []

    for cl in my_tree_clades:
        X.append(x_coords[cl])
        Y.append(y_coords[cl])
        text.append(cl.name)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=X,
                y=Y,
                mode="markers",
                text=text,
                marker=dict(color="rgb(100,100,100)", size=5),
                hoverinfo="text",
            )
        ],
        layout=go.Layout(
            title="Single linkage tree",
            dragmode="select",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=True, zeroline=False, showticklabels=True, title="Branch Length"),
            yaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False),
            shapes=line_shapes,  # Add line shapes to the layout
        )
    )

    return fig


def get_x_coordinates(tree):
    """Associates to  each clade an x-coord.
       returns dict {clade: x-coord}
    """
    xcoords = tree.depths()
    # tree.depth() maps tree clades to depths (by branch length).
    # returns a dict {clade: depth} where clade runs over all Clade instances of the tree, and depth
    # is the distance from root to clade

    #  If there are no branch lengths, assign unit branch lengths
    if not max(xcoords.values()):
        xcoords = tree.depths(unit_branch_lengths=True)
    return xcoords


def get_y_coordinates(tree, dist=1.3):
    """
       returns  dict {clade: y-coord}
       The y-coordinates are  (float) multiple of integers (i*dist below)
       dist depends on the number of tree leafs
    """
    maxheight = tree.count_terminals()  # Counts the number of tree leafs.
    # Rows are defined by the tips/leafs
    ycoords = dict(
        (leaf, maxheight - i * dist)
        for i, leaf in enumerate(reversed(tree.get_terminals()))
    )

    def calc_row(clade):
        for subclade in clade:
            if subclade not in ycoords:
                calc_row(subclade)
        ycoords[clade] = (ycoords[clade.clades[0]] + ycoords[clade.clades[-1]]) / 2

    if tree.root.clades:
        calc_row(tree.root)
    return ycoords


def draw_clade(
    clade,
    x_start,
    line_shapes,
    line_color="rgb(15,15,15)",
    line_width=1,
    x_coords=0,
    y_coords=0,
):
    """Recursively draw the tree branches, down from the given clade"""

    x_curr = x_coords[clade]
    y_curr = y_coords[clade]

    # Draw a horizontal line from start to here
    branch_line = get_clade_lines(
        orientation="horizontal",
        y_curr=y_curr,
        x_start=x_start,
        x_curr=x_curr,
        line_color=line_color,
        line_width=line_width,
    )

    line_shapes.append(branch_line)

    if clade.clades:
        # Draw a vertical line connecting all children
        y_top = y_coords[clade.clades[0]]
        y_bot = y_coords[clade.clades[-1]]

        line_shapes.append(
            get_clade_lines(
                orientation="vertical",
                x_curr=x_curr,
                y_bot=y_bot,
                y_top=y_top,
                line_color=line_color,
                line_width=line_width,
            )
        )

        # Draw descendants
        for child in clade:
            draw_clade(child, x_curr, line_shapes, x_coords=x_coords, y_coords=y_coords)


def get_clade_lines(
    orientation="horizontal",
    y_curr=0,
    x_start=0,
    x_curr=0,
    y_bot=0,
    y_top=0,
    line_color="rgb(25,25,25)",
    line_width=0.5,
):
    """define a shape of type 'line', for branch
    """
    branch_line = dict(
        type="line", layer="below", line=dict(color=line_color, width=line_width)
    )
    if orientation == "horizontal":
        branch_line.update(x0=x_start, y0=y_curr, x1=x_curr, y1=y_curr)
    elif orientation == "vertical":
        branch_line.update(x0=x_curr, y0=y_bot, x1=x_curr, y1=y_top)
    else:
        raise ValueError("Line type can be 'horizontal' or 'vertical'")

    return branch_line



if __name__=="__main__":
    newick_tree = "((A:1,B:1):2,(C:1,D:1):2);"
    fig = create_tree(newick_tree)
    fig.write_image("datasets/dendrogram_with_lines.png")