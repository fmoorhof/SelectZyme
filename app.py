"""todo: make landing page here and then reference the different pages. somehow share the table components across tables (possible?)
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import networkx as nx

import pages.mst as mst
import pages.single_linkage as sl


# Initialize the Dash app
app = dash.Dash(
    __name__,
    use_pages=True,  # Enables the multi-page functionality
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],  # Optional for styling
)
server = app.server

# Define the graph G
G = nx.random_geometric_graph(10, 0.5, seed=42)
dash.register_page('mst', name="Minimal Spanning Tree", layout=mst.layout(G))
dash.register_page('single-linkage', layout=sl.layout(G))

# Layout with navigation links and page container
app.layout = dbc.Container(
    [
        dbc.NavbarSimple(
            brand="Please see here the Analysis results",
            color="primary",
            dark=True,
        ),
        html.Div(
            [
                dbc.Nav(
                    [
                        dbc.NavItem(
                            dbc.NavLink(page["name"], href=page["path"])
                        )
                        for page in dash.page_registry.values()
                    ],
                    # pills=True,
                ),
                # html.Hr(),
                dash.page_container,  # Displays the content of the current page
            ]
        ),
    ],
    fluid=True,
)

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
