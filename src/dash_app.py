"""Definitions for the Dash app.

Dash wiki was insanely helpful for the DataTable. If you like to implement additional features, check out the documentation:
https://dash.plotly.com/datatable/editable"""
import logging
 
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px


def run_dash_app(df, X_red, method: str, project_name: str, app: dash.Dash):
    """Run a Dash app to visualize the results of the dimensionality reduction.
    app.layout is setting my custom layout with plotly express.
    app.callback is setting the callback function to show the data of the selected points in the plot. 
    update_table is the callback function, filling the table with the data of the selected points.
    
    :param df: dataframe containing the annoattions
    :param X_red: dimensionality reduced embeddings
    :param method: dimensionality reduction method used
    :param project_name: name of the collection/dataset
    :param app: dash app
    return: dash app"""
    cols = df.columns.values.tolist()
    # Define the layout of the Dash app
    # app = dash.Dash(__name__)
    app.layout = html.Div([
        dcc.Graph(
            id='plot',
            figure=px.scatter(df,
                            x=X_red[:, 0],
                            y=X_red[:, 0],
                            color='EC number',  # color='cluster'
                            title=f'2D {method} on dataset {project_name}',
                            hover_data=cols,
                            opacity=0.2,
                            color_continuous_scale=px.colors.cyclical.Edge,  # px.colors.sequential.Viridis,
                            symbol=df['marker_symbol'],
                            size=df['marker_size'],
        ),        
            config={
                'scrollZoom': True,      
            },
            style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
        ),
        # Define appearance of the table
        dash_table.DataTable(
        id='data-table',
        columns=[{'id': c, 'name': c} for c in df.columns],
        style_cell={'textAlign': 'left'},
        editable=True,
        row_deletable=True,
        export_format='xlsx',
        export_headers='display',
        merge_duplicate_headers=True,
        # additional stuff:
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        column_selectable="single",
        row_selectable="multi",
    )
    ])

    # Define the callback function
    @app.callback(
        Output('data-table', 'data'),
        Input('plot', 'clickData'),
        [dash.dependencies.State('data-table', 'data')]
    )

    def update_table(clickData, existing_table):
        if clickData is None:
            return existing_table
    
        selected_feature = clickData['points'][0]['customdata']
        logging.info(selected_feature)
        print(selected_feature)
    
        # Create a new row for the table
        new_row = {col: value for col, value in zip(df.columns, selected_feature)}
    
        # If the existing_table is None (first time), use an empty table
        if existing_table is None:
            existing_table = []
    
        # Append the new row to the existing table
        return existing_table + [new_row]
    return app
 
 
if __name__ == '__main__':
    NotImplementedError('This script is not meant to be run as main.')
    # app.run_server(debug=True, port=8050)  # make available to localhost    
    # app.run_server(host='0.0.0.0', port=8050, debug=False)  # from ocean to local machine