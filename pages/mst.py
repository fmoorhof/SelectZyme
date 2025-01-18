from base64 import b64encode
import io

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

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
    mst = MinimumSpanningTree(G._mst, G._data, X_red, df)
    fig = mst.plot_mst_in_DimRed_landscape()
    fig = mst.plot_mst_force_directed(G)

    layout = html.Div([
        # plot download button
        html.Div(
            html.A(
            html.Button("Download plot as HTML"), 
            id="download",
            href=_html_export_figure(fig),  # if other column got selected see callback (update_plot_and_download) for export definition
            download="plotly_graph.html"
            ),
            style={'float': 'right', 'display': 'inline-block'}
        ),

        # Scatter plot
        dcc.Graph(
            id='plot',
            figure=fig,        
            config={'scrollZoom': True,},
            style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
        ),

        # data table
        dash_table.DataTable(
            id='data-table',
            columns=[{'id': c, 'name': c} for c in df.columns],
            style_cell={
                'textAlign': 'left',
                'maxWidth': '200px',  # Set a maximum width for all columns
                'whiteSpace': 'normal',  # Allow text to wrap within cells
                'overflow': 'hidden',  # Hide overflow content
                'textOverflow': 'ellipsis',  # Add ellipsis for overflow text
                },
            style_data={
                'width': '150px',  # Set a fixed width for data cells
            },
            style_table={
                'maxWidth': '100%',  # Set the table width to 100% of its container
                'overflowX': 'auto',  # Enable horizontal scrolling
            },
            editable=True,
            row_deletable=True,
            export_format='xlsx',
            export_headers='display',
            merge_duplicate_headers=True,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            row_selectable="multi",
        )
    ])  # closing html.Div finally

    return layout


def register_callbacks(app, df):
    # Define callbacks
    @app.callback(
        Output('data-table', 'data'),
        Input('plot', 'clickData'),
        dash.dependencies.State('data-table', 'data')
    )
    def update_table(clickData, existing_table):
        if clickData is None:
            return existing_table

        # extract accession from selection and lookup row in df and append row to the dash table
        accession  = clickData['points'][0]['customdata']  # accession  = clickData['points'][0]['text'].split('<br>')[0].replace('accession: ', '')  # if customdata fails 
        selected_row = df[df['accession'] == accession].iloc[0]
        selected_row[df.columns.get_loc('selected')] = True  # if entry has been selected once set it to True
        # build Brenda URLs
        if selected_row['xref_brenda'] != '':
            selected_row['BRENDA URL'] = f"https://www.brenda-enzymes.org/enzyme.php?ecno={selected_row['xref_brenda'].split(';')[0]}&UniProtAcc={selected_row['accession']}&OrganismID={selected_row['organism_id']}"

        if existing_table is None:
            existing_table = []
        existing_table.append(selected_row.to_dict())

        return existing_table
    

def _html_export_figure(fig):
        buffer = io.StringIO()
        fig.write_html(buffer)
        html_bytes = buffer.getvalue().encode()
        encoded = b64encode(html_bytes).decode()
        return f"data:text/html;base64,{encoded}"    