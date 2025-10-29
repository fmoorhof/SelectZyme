from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html
from plotly.graph_objects import Figure

from selectzyme.pages.callbacks import html_export_figure

# Register page with custom URL path, must be done in app.py if app.layout is in a function layout
# dash.register_page(__name__, path="/dim", name="DimRed")


def layout(columns: list, fig: Figure, dropdown = False) -> html.Div:
    """
    This layout includes:
    - An OPTIONAL dropdown menu for selecting a legend attribute from the provided columns.
    - A button to download the interactive plot as an HTML file.
    - A scatter plot displayed with a loading spinner.
    - A data table displaying the provided columns along with additional columns for "x", "y", and "BRENDA URL".
    Args:
        columns (list): A list of column names to be displayed in the dropdown and data table. 
                        The "accession" column is removed from this list.
        fig (Figure): A Plotly figure object to be displayed in the scatter plot.
        dropdown (bool): A flag to indicate whether to show the dropdown menu for selecting a legend attribute.
    Returns:
        html.Div: A Dash HTML Div containing the layout components.
    """
    # columns.remove("accession")
    return html.Div(
        [
            # Dropdown to select legend attribute of df columns. ONLY shown if dropdown is True
            html.Div(
                [
                    html.Label("Select legend attribute", htmlFor="legend-attribute"),
                    dbc.Select(
                        id="legend-attribute",
                        options=[{"label": col, "value": col} for col in columns],
                        value=columns[-1],  # default selection
                    ),
                ],
                style={"width": "30%", "display": "inline-block", "verticalAlign": "top"} if dropdown else {"display": "none"},
            ),
            # plot and FASTA download buttons
            html.Div(
                [
                    html.A(
                        html.Button("Download interactive plot"),
                        id="download-button",
                        href=html_export_figure(fig),
                        download="plotly_graph.html",
                    ),
                    html.A(
                        html.Button("Download table as FASTA"),
                        id="download-fasta-button",
                        href="/download/fasta",
                        download="data.fasta",
                        style={"marginLeft": "10px"},
                    ),
                ],
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
                fullscreen=False,  # for loading message
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
                    "maxWidth": "200px",
                    "whiteSpace": "normal",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
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
                # filter_action="native",
                sort_action="native",
                sort_mode="multi",
                column_selectable="single",
                row_selectable="multi",
            ),
        ],
            )
