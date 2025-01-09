"""missing todo: parse df annotations to nodes of tree"""
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import networkx as nx
import pandas as pd
import plotly.figure_factory as ff

from src.hdbscan_plotting import SingleLinkageTree
from src.phylogenetic_tree import g_to_newick, create_tree, create_tree_circular
# from src.phylogenetic_tree_circular import circos_dendrogram


def layout(G, df: pd.DataFrame) -> html.Div:
    fig = SingleLinkageTree(G._linkage, df).plot()  # latest implementation sing the hdbscan_plotting
    
    newick_str = g_to_newick(G.to_networkx())
    # fig = create_tree_circular(newick_str)  # create_tree(newick_str)

    # attempt with the plotly figure factory: dendrogram front-end rendering ultra slow. browser freezes!
    # fig = ff.create_dendrogram(G._linkage)  # todo: labels=df

    # attempt circos_dendrogram: figure creation ultra slow. -> not usable
    # fig = circos_dendrogram(newick_str)  # todo: fig needs to be plotly figure not matplotlib
    # fig.savefig("datasets/pycirclize.png")

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
