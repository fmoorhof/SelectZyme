"""Implementations inspired and mostly taken from: https://github.com/plotly/dash-sample-apps/blob/main/apps/dash-phylogeny/utils.py
networkx graph to newick format implementation is taken from: https://stackoverflow.com/questions/46444454/save-networkx-tree-in-newick-format"""
import io

import networkx as nx
from Bio import Phylo
import plotly.graph_objs as go


def g_to_newick_malfunction(g, root=None):
    """
    todo: fix malfunction: root node not found. is it because of example data or mal-implementation?
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
        roots = list(filter(lambda p: p[1] == 0, g.degree()))
        assert 1 == len(roots)
        root = roots[0][0]
    subgs = []
    for child in g[root]:
        if len(g[child]) > 0:
            subgs.append(g_to_newick_malfunction(g, root=child))
        else:
            subgs.append(child)
    return "(" + ','.join(subgs) + ")"


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
            title="Minimal Tree Plot",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=True, zeroline=False, showticklabels=True, title="Branch Length"),
            yaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False),
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


def recursive_search(dict, key):
    """
    Recursively searches for a key in a nested dictionary and returns its value if found.
    Args:
        dict (dict): The dictionary to search through. It can contain nested dictionaries.
        key (str): The key to search for in the dictionary.
    Returns:
        The value associated with the key if found, otherwise None.
    """

    if key in dict:
        return dict[key]
    for k, v in dict.items():
        item = recursive_search(v, key)
        if item is not None:
            return item


def bfs_edge_lst(graph, n):
    """
    Perform a breadth-first search (BFS) on a graph starting from a given node and return the list of edges.
    Parameters:
    graph (networkx.Graph): The graph on which to perform BFS.
    n (node): The starting node for the BFS.
    Returns:
    list: A list of edges in the order they were traversed in the BFS.
    """
    return list(nx.bfs_edges(graph, n))


def tree_from_edge_lst(elst, n):
    """
    Constructs a tree from a list of edges.
    Args:
        elst (list of tuples): A list of edges where each edge is represented as a tuple (src, dst).
        n (int or str): The root node of the tree.
    Returns:
        dict: A nested dictionary representing the tree structure.
    """

    tree = {n: {}}
    for src, dst in elst:
        subt = recursive_search(tree, src)
        subt[dst] = {}
    return tree


def tree_to_newick(tree):
    """
    Convert a phylogenetic tree represented as a nested dictionary to a Newick format string.
    Args:
        tree (dict): A nested dictionary representing the phylogenetic tree. Each key is a node, and its value is another dictionary representing its children.
    Returns:
        str: A string representing the tree in Newick format.
    """
    items = []
    for k in tree.keys():
        s = ''
        if len(tree[k].keys()) > 0:
            subt = tree_to_newick(tree[k])
            if subt != '':
                s += '(' + subt + ')'
        s += k
        items.append(s)
    return ','.join(items)


def g_to_newick(g):
    """
    Convert a graph to Newick format.
    This function takes a graph `g` and converts it to a Newick formatted string.
    The graph is assumed to have '1' as the root node.
    Args:
        g (networkx.Graph): The input graph to be converted.
    Returns:
        str: The Newick formatted string representation of the graph.
    """

    elst = bfs_edge_lst(g, '1')  #'1' being the root node of the graph
    tree = tree_from_edge_lst(elst, '1')
    newick = tree_to_newick(tree) + ';'

    return newick