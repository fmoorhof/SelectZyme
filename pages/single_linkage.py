"""missing todo: parse df annotations to nodes of tree"""
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import networkx as nx
import pandas as pd

from src.phylogenetic_tree import create_tree, g_to_newick


def layout(G: nx.Graph, df: pd.DataFrame) -> html.Div:
    newick_str = g_to_newick(G)
    fig = create_tree(newick_str)  # fig is created externally

    # Layout
    layout = html.Div(
        [
            dcc.Graph(id="linkage-plot", figure=fig),
            dash_table.DataTable(
                id="data-table",
                columns=[{"id": col, "name": col} for col in df.columns],
                data=[],  # Initially empty
            ),
        ]
    )

    # Callbacks, populate table data
    # @dash.callback(
    #     Output("data-table", "data"),
    #     Input("network-plot", "clickData"),
    # )
    # def update_table(clickData):
    #     if clickData is None:
    #         return []

    #     selected_node = int(clickData["points"][0]["text"])
    #     row = df[df["node_id"] == selected_node]
    #     return row.to_dict("records")

    return layout
