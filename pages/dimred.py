from dash import dcc, html, dash_table

from src.customizations import set_columns_of_interest
from pages.callbacks import html_export_figure


def layout(df, fig):
    """
    Generate the layout for a Dash app with a 2D plot, dropdown for selecting legend attribute, 
    download button, scatter plot, and data table.
    Parameters:
    df (pd.DataFrame): DataFrame containing the data to be visualized.
    X_red (np.ndarray): Reduced dimensionality data for plotting.
    X_red_centroids (np.ndarray): Centroid data for the reduced dimensionality data.
    Returns:
    html.Div: A Dash HTML Div containing the layout of the app.
    """
    cols = set_columns_of_interest(df.columns)
    cols.remove('accession')
    return html.Div(
        [
            # Dropdown to select legend attribute of df columns
            html.Div(
                [
                    # Plot display selector
                    dcc.Dropdown(
                        id='legend-attribute',
                        options=[
                            {'label': col, 'value': col} for col in cols
                        ],
                        value=cols[0],  # set default column to show on loading
                    )
                ],
                style={'width': '30%', 'display': 'inline-block'},
            ),
            # plot download button
            html.Div(
                html.A(
                    html.Button("Download plot as HTML"),
                    id="download-button",
                    href=html_export_figure(fig),  # if other column got selected see callback (update_plot_and_download) for export definition
                    download="plotly_graph.html",
                ),
                style={'float': 'right', 'display': 'inline-block'},
            ),
            # Scatter plot
            dcc.Graph(
                id='plot',
                figure=fig,
                config={
                    'scrollZoom': True,
                },
                style={
                    'width': '100%',
                    'height': '100%',
                    'display': 'inline-block',
                },
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
            ),
        ]
    )