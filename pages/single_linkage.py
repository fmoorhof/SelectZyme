"""missing todo: parse df annotations to nodes of tree"""
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import networkx as nx
import pandas as pd

from src.hdbscan_plotting import SingleLinkageTree
# from src.phylogenetic_tree import g_to_newick  #, create_tree
# from src.phylogenetic_tree_circular import create_tree_circular as create_tree


def layout(G, df: pd.DataFrame) -> html.Div:
    fig = SingleLinkageTree(G._linkage, df).plot()
    # newick_str = g_to_newick(G)
    # fig = create_tree(newick_str)  # fig is created externally

    # Layout
    layout = html.Div(
        [
            dcc.Graph(id="linkage-plot", 
                      figure=fig,
                      config={'scrollZoom': True},
                      style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
                      ),
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
    #     Input("linkage-plot", "clickData"),
    # )
    # def update_table(clickData):
    #     if clickData is None:
    #         return []

    #     selected_node = int(clickData["points"][0]["text"])
    #     row = df[df["node_id"] == selected_node]
    #     return row.to_dict("records")

    return layout
