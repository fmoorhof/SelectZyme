from base64 import b64encode
import io

from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

from visualizer import plot_2d


# def layout(df, X_red):        
#     layout, register_callbacks = run_dash_app(df, X_red)
#     return layout, register_callbacks


def layout(df, X_red, X_red_centroids):
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
    cols = df.columns.values.tolist()

    fig = plot_2d(df, X_red, X_red_centroids, legend_attribute=cols[2])

    # Define the layout of the Dash app
    layout = html.Div([
        # Dropdown to select legend attribute of df columns
        html.Div([
            # Plot display selector
            dcc.Dropdown(
                id='legend-attribute',
                options = [{'label': col, 'value': col} for col in cols],  # cols[:12]
                value=cols[2]  # set default column to show on loading
            )], 
            style={'width': '30%', 'display': 'inline-block'}
            ),
        
        # plot download button
        html.Div(
            html.A(
            html.Button("Download plot as HTML"), 
            id="download-button",
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


def register_callbacks(app, df, X_red, X_red_centroids):
    # Define callbacks
    @app.callback(
        [Output('data-table', 'data'),  # output data to table displayed at page
         Output('shared-data', 'data')],  # output data to update the dcc.Store in app.py
        [Input('plot', 'clickData'),  # input users selection/click data
         Input('shared-data', 'data')],  # input from dcc.Store in app.py (to load existing table at another page)
        State('data-table', 'data'),  # !if user edits the table (delete rows, edit cells), changes are saved here
        # State('shared-data', 'data')
    )
    def update_table(clickData, shared_table, data_table):  # (input 1, input 2, state)
        """
        Updates the existing table with a new row based on the clickData.
        Parameters:
        clickData (dict): Data from a click event, containing information about the selected point.
        existing_table (list): The current state of the table, represented as a list of dictionaries.
        Returns:
        list: The updated table with the new row appended.
        Notes:
        - If clickData is None, the function returns the existing table without any changes.
        - The function extracts the accession from the clickData and looks up the corresponding row in the dataframe `df`.
        - The selected row is marked as selected by setting the 'selected' column to True.
        - If the selected row has a non-empty 'xref_brenda' field, a BRENDA URL is constructed and added to the row.
        - If the existing table is None, it initializes it as an empty list.
        - The selected row is converted to a dictionary and appended to the existing table.
        """
        if clickData is None:
            return shared_table  # fix: not working, still table loads empty until 1st click (remove next line then, too)
        
        # if user deletes entries or modifies cells
        if data_table is not None:  # skip reload events/page changes since data_table is None (remove with fix above when done)
             if data_table != shared_table:
                shared_table = data_table  # set modified changes of user to shared_table (dcc.Store)

        # extract accession from selection and lookup row in df and append row to the dash table
        accession  = clickData['points'][0]['customdata']  # accession  = clickData['points'][0]['text'].split('<br>')[0].replace('accession: ', '')  # if customdata fails 
        selected_row = df[df['accession'] == accession].iloc[0]
        selected_row[df.columns.get_loc('selected')] = True  # if entry has been selected once set it to True
        
        # build Brenda URLs
        if selected_row['xref_brenda'] != '':
            selected_row['BRENDA URL'] = f"https://www.brenda-enzymes.org/enzyme.php?ecno={selected_row['xref_brenda'].split(';')[0]}&UniProtAcc={selected_row['accession']}&OrganismID={selected_row['organism_id']}"

        if shared_table is None:
            shared_table = []
        shared_table.append(selected_row.to_dict())

        return shared_table, shared_table

    @app.callback(
        [Output('plot', 'figure'), Output('download-button', 'href')],
        Input('legend-attribute', 'value')
    )
    def update_plot_and_download(legend_attribute):
        """
        Updates the plot and generates a download link for the updated figure.
        Args:
            legend_attribute (str): The attribute to be used for the legend in the plot.
        Returns:
            tuple: A tuple containing the updated figure and the updated download link.
                - updated_fig: The updated 2D plot figure.
                - updated_href: The HTML href link for downloading the updated figure.
        """
        updated_fig = plot_2d(df, X_red, X_red_centroids, legend_attribute)  # Update the figure
        updated_href = _html_export_figure(updated_fig)  # Generate the updated download link
        return updated_fig, updated_href
    

def _html_export_figure(fig):
        """
        Converts a Plotly figure to an HTML string and encodes it in base64 format.
        Args:
            fig (plotly.graph_objs._figure.Figure): The Plotly figure to be converted.
        Returns:
            str: A base64 encoded HTML string representing the figure.
        """
        buffer = io.StringIO()
        fig.write_html(buffer)
        html_bytes = buffer.getvalue().encode()
        encoded = b64encode(html_bytes).decode()
        return f"data:text/html;base64,{encoded}" 