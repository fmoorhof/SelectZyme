"""Deprecation warning: This file can be deleted if i stick to the hdbscan_plotting implementation!

Neither normal nor circular are implemented well. tree spans over the whole plot 1 branch each so the implementation is utterly wrong. remove

Implementations inspired and mostly taken from: https://github.com/plotly/dash-sample-apps/blob/main/apps/dash-phylogeny/utils.py
networkx graph to newick format implementation is taken from: https://stackoverflow.com/questions/46444454/save-networkx-tree-in-newick-format
"""
import io
import math

from Bio import Phylo
import plotly.graph_objs as go


def g_to_newick(G, root=None):
    """
    Convert a directed graph to Newick format, including node attributes.

    Parameters:
    G (networkx.DiGraph): A directed graph representing the tree.
    root (node, optional): The root node of the tree. If None, the root is determined automatically.

    Returns:
    str: A string representing the tree in Newick format, including node attributes.

    Raises:
    AssertionError: If the graph does not have exactly one root node.
    """
    if root is None:
        # Determine the root as a node with no incoming edges
        roots = list(filter(lambda p: p[1] == 0, G.in_degree()))
        assert len(roots) == 1, "Graph must have exactly one root node."
        root = roots[0][0]
    
    subgs = []
    for child in G.successors(root):  # Use successors for directed graphs
        if G.out_degree(child) > 0:  # If the node has children
            subgs.append(g_to_newick(G, root=child))
        else:  # Leaf node
            subgs.append(node_to_newick(G, child))
    
    root_label = node_to_newick(G, root)
    return f"({','.join(subgs)}){root_label}"


def node_to_newick(G, node):
    """
    Generate the Newick label for a node, including its attributes.

    Parameters:
    G (networkx.DiGraph): The graph containing the node.
    node (hashable): The node for which to generate the label.

    Returns:
    str: A Newick-compatible string for the node.
    """
    # Start with the node's name or ID
    node_label = str(node)

    # If the node has attributes, append them in Newick-compatible format
    attributes = G.nodes[node]
    if attributes:  # Only include attributes if they exist
        attr_str = ",".join(f"{key}={value}" for key, value in attributes.items())
        node_label += f"[{attr_str}]"
    
    return node_label


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
        if cl.name:  # revert this if G.attributes shall be read from G instead str(newick_tree)
            text.append(cl.comment)
        else: 
            text.append(cl.name)  # non leaf nodes are not named

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


# circular implementation WIP: layout looks not nice (not really circular, looks unprofessional)
def create_tree_circular(nw_tree):
    tree = Phylo.read(io.StringIO(nw_tree), "newick")
    polar_coords = assign_polar(tree)
    line_shapes = []

    draw_clade_circular(
        tree.root,
        line_shapes,
        polar_coords=polar_coords,
        line_color="rgb(25,25,25)",
        line_width=1,
    )

    X = [radius * math.cos(theta) for radius, theta in polar_coords.values()]
    Y = [radius * math.sin(theta) for radius, theta in polar_coords.values()]
    text = [clade.name for clade in polar_coords.keys()]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=X,
                y=Y,
                mode="markers+text",
                text=text,
                textposition="top center",
                marker=dict(color="rgb(100,100,100)", size=5),
                hoverinfo="text",
            )
        ],
        layout=go.Layout(
            title="Circular Phylogenetic Tree",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            shapes=line_shapes,
        ),
    )

    return fig


def draw_clade_circular(
    clade,
    line_shapes,
    polar_coords,
    parent=None,
    line_color="rgb(15,15,15)",
    line_width=1,
):
    """Recursively draw the tree branches in circular layout."""
    radius_curr, theta_curr = polar_coords[clade]

    # Draw a line to the parent node if it exists
    if parent is not None:
        radius_parent, theta_parent = polar_coords[parent]
        branch_line = get_clade_lines_circular(
            theta_start=theta_parent,
            theta_curr=theta_curr,
            radius_start=radius_parent,
            radius_curr=radius_curr,
            line_color=line_color,
            line_width=line_width,
        )
        line_shapes.append(branch_line)

    # Recursively process child clades
    for child in clade.clades:
        draw_clade_circular(
            child,
            line_shapes,
            polar_coords=polar_coords,
            parent=clade,  # Pass the current clade as the parent for the child
            line_color=line_color,
            line_width=line_width,
        )


def get_clade_lines_circular(
    theta_start, theta_curr, radius_start, radius_curr, line_color, line_width
):
    """Define a shape of type 'path' for a curved branch in circular layout."""
    # Convert polar to Cartesian coordinates
    x_start = radius_start * math.cos(theta_start)
    y_start = radius_start * math.sin(theta_start)
    x_end = radius_curr * math.cos(theta_curr)
    y_end = radius_curr * math.sin(theta_curr)

    # Create a path for the curved branch
    path = f"M {x_start},{y_start} L {x_end},{y_end}"
    return dict(
        type="path",
        path=path,
        line=dict(color=line_color, width=line_width),
        layer="below",
    )


def assign_polar(tree, radius_increment=1.0):
    """Assign polar coordinates (radius, theta) to all nodes."""
    polar_coords = {}
    leaf_count = len(tree.get_terminals())
    angle_increment = 2 * math.pi / leaf_count

    def set_coords(clade, depth=0, angle_start=0, angle_end=2 * math.pi):
        if clade.is_terminal():
            theta = (angle_start + angle_end) / 2
            radius = depth * radius_increment
            polar_coords[clade] = (radius, theta)
        else:
            angle_span = (angle_end - angle_start) / len(clade.clades)
            for i, child in enumerate(clade.clades):
                child_angle_start = angle_start + i * angle_span
                child_angle_end = child_angle_start + angle_span
                set_coords(child, depth + 1, child_angle_start, child_angle_end)
            # For internal nodes, use the midpoint of their children's coordinates
            child_coords = [polar_coords[child] for child in clade.clades]
            avg_radius = depth * radius_increment
            avg_theta = sum(theta for _, theta in child_coords) / len(child_coords)
            polar_coords[clade] = (avg_radius, avg_theta)

    set_coords(tree.root)
    return polar_coords



if __name__=="__main__":
    newick_tree = "(((A:1,B:1):2,(C:1,D:1):2):3,((E:1,F:1):2,(G:1,H:1):2):3);"
    fig = create_tree(newick_tree)
    fig.write_image("datasets/dendrogram_with_lines.png")

    fig = create_tree_circular(newick_tree)
    # fig.write_image("datasets/dendrogram_circular.png")