import dash
from dash import html, dcc

dash.register_page(__name__, path="/single_linkage")  # Register page with custom URL path

def layout():
    return html.Div(
        [
            html.H1("Single Linkage Tree App"),
            dcc.Graph(id="single-linkage-plot"),  # Add your plot here
            dcc.Store(id="single-linkage-data"),  # Optional for storing data
        ]
    )

# Add callbacks here for Single Linkage functionality
