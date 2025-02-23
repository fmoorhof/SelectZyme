from __future__ import annotations

import logging

from dash import dash_table, dcc, html

from pages.dimred import html_export_figure
from src.mst_plotting import MinimumSpanningTree

# Register page
# dash.register_page(__name__, path="/mst", name="Minimal Spanning Tree")  # Register page with custom URL path, must be done in app.py if app.layout is in a function layout


def layout(G, df, X_red, fig) -> html.Div:
    """
    Generates a Dash layout for visualizing a minimal spanning tree of a given graph.
    Parameters:
    G (nx.Graph): The input graph for which the minimal spanning tree layout is to be generated.
    df (pd.DataFrame): A DataFrame containing node information such as node ID, x and y positions, and number of connections.
    Returns:
    html.Div: A Dash HTML Div containing the graph visualization and a data table.
    The function performs the following steps:
    1. Computes the spring layout positions for the nodes in the graph.
    2. Sets the node positions as attributes in the graph.
    3. Modifies the graph data to create edge and node traces for visualization.
    4. Creates a figure dictionary for the graph visualization.
    5. Creates a DataFrame containing node information such as node ID, x and y positions, and number of connections.
    6. Constructs a Dash layout with a graph component and a data table.
    7. Defines a callback to update the data table based on node clicks in the graph.
    Note:
    - The `modify_graph_data` function is assumed to be defined elsewhere and is responsible for creating the edge and node traces.
    """
    logging.info("Start building the MST...")
    mst = MinimumSpanningTree(G._mst, df, X_red, fig)

    if df.shape[0] > 1:
        fig = mst.plot_mst_in_dimred_landscape()
    else:
        fig = mst.plot_mst_force_directed(G)

    return html.Div(
        [
            # plot download button
            html.Div(
                html.A(
                    html.Button("Download plot as HTML"),
                    id="download-button",
                    href=html_export_figure(
                        fig
                    ),  # if other column got selected see callback (update_plot_and_download) for export definition
                    download="plotly_graph.html",
                ),
                style={"float": "right", "display": "inline-block"},
            ),
            # Scatter plot
            dcc.Graph(
                id="plot",
                figure=fig,
                config={
                    "scrollZoom": True,
                },
                style={
                    "width": "100%",
                    "height": "100%",
                    "display": "inline-block",
                },
            ),
            # data table
            dash_table.DataTable(
                id="data-table",
                columns=[{"id": c, "name": c} for c in df.columns],
                style_cell={
                    "textAlign": "left",
                    "maxWidth": "200px",  # Set a maximum width for all columns
                    "whiteSpace": "normal",  # Allow text to wrap within cells
                    "overflow": "hidden",  # Hide overflow content
                    "textOverflow": "ellipsis",  # Add ellipsis for overflow text
                },
                style_data={
                    "width": "150px",  # Set a fixed width for data cells
                },
                style_table={
                    "maxWidth": "100%",  # Set the table width to 100% of its container
                    "overflowX": "auto",  # Enable horizontal scrolling
                },
                editable=True,
                row_deletable=True,
                export_format="xlsx",
                export_headers="display",
                merge_duplicate_headers=True,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                column_selectable="single",
                row_selectable="multi",
            ),
        ]
    )
