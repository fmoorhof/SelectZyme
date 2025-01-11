import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import networkx as nx
import pandas as pd

from src.dash_app_network import modify_graph_data
from src.hdbscan_plotting import MinimumSpanningTree


# Register page
# dash.register_page(__name__, path="/mst", name="Minimal Spanning Tree")  # Register page with custom URL path, must be done in app.py if app.layout is in a function layout


def layout(G, df, X_red) -> html.Div:
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
    # define graph layout and coordinates
    # pos = nx.spring_layout(G)
    # pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", root=0)  # Warning: specified root node "0" was not found.Using default calculation for root node
    # nx.set_node_attributes(G, pos, 'pos')

    # edge_trace, node_trace = modify_graph_data(G)

    # # Create figure
    # fig = {
    #     "data": [edge_trace, node_trace],
    #     "layout": dict(
    #         hovermode="closest",
    #         margin=dict(b=20, l=5, r=5, t=40),
    #         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    #         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    #     ),
    # }

    # hdbscan_plotting implementation
    mst = MinimumSpanningTree(G._mst, G._data, X_red, df)
    fig = mst.plot()

    # Layout
    layout = html.Div(
        [
            dcc.Graph(id="mst-plot", 
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
    #     Input("mst-plot", "clickData"),
    # )
    # def update_table(clickData):
    #     if clickData is None:
    #         return []

    #     selected_node = int(clickData["points"][0]["text"])
    #     row = df[df["node_id"] == selected_node]
    #     return row.to_dict("records")

    return layout
