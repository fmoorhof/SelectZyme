from __future__ import annotations

import logging

import pandas as pd
from dash import dash_table, dcc, html

from pages.dimred import html_export_figure
from selectzyme.single_linkage_plotting import create_dendrogram


def layout(G, df: pd.DataFrame, legend_attribute: str, out_file: str) -> html.Div:
    logging.info("Start building the dendrogram...")

    fig = create_dendrogram(
        Z=G._linkage, df=df, legend_attribute=legend_attribute
    )
    fig.write_html(out_file)

    return html.Div(
        [
            # plot download button
            html.Div(
                html.A(
                    html.Button("Download plot as HTML"),
                    id="download-button",
                    href=html_export_figure(
                        fig
                    ),
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
                columns=[{"id": c, "name": c} for c in df.columns] + [{"id": "x", "name": "x"}, {"id": "y", "name": "y"}, {"id": "BRENDA URL", "name": "BRENDA URL"}],
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
