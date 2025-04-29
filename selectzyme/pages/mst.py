from __future__ import annotations

from dash import dash_table, dcc, html
from plotly.graph_objects import Figure

from selectzyme.pages.dimred import html_export_figure

# Register page with custom URL path, must be done in app.py if app.layout is in a function layout
# dash.register_page(__name__, path="/mst", name="Minimal Spanning Tree")


def layout(columns: list, fig: Figure) -> html.Div:
    """
    Generates a Dash layout for visualizing a minimal spanning tree of a given graph.
    Parameters:
    columns (list): List of column names to be displayed in the data table.
    fig (go.Figure): A Plotly figure object for the minimal spanning tree.
    """
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
            # Scatter plot with loading message
            dcc.Loading(
                id="loading-plot",
                type="default",  # or "circle", "dot", "cube"
                children=dcc.Graph(
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
                fullscreen=False,
            ),
            html.Div(
                id="loading-text",
                style={"textAlign": "center", "marginTop": "10px", "fontStyle": "italic"},
            ),
            # data table
            dash_table.DataTable(
                id="data-table",
                columns=[{"id": c, "name": c} for c in columns] + [{"id": "x", "name": "x"}, {"id": "y", "name": "y"}, {"id": "BRENDA URL", "name": "BRENDA URL"}],
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
