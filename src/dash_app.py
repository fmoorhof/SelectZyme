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
        # Dropdown to select legend attribute of df columns
        html.Div([
            dcc.Dropdown(
                id='legend-attribute',
                options = [{'label': col, 'value': col} for col in cols],  # cols[:12]
                value=cols[2]
                # options=[{'label': 'Cluster', 'value': 'cluster'},],
                # value='cluster'  # Default color by 'cluster'
            )
        ], style={'width': '30%', 'display': 'inline-block'}),

        # Scatter plot
        dcc.Graph(
            id='plot',
            figure=px.scatter(df,
                            x=X_red[:, 0],
                            y=X_red[:, 1],
                            color='ec',  # color='cluster'
                            title=f'2D {method} on dataset {project_name}',
                            hover_data=cols,
                            opacity=0.4,
                            color_continuous_scale=px.colors.sequential.Viridis,  # px.colors.sequential.Viridis, px.colors.cyclical.Edge
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
        """This function updates the table with the data of the selected points in the plot. There is several functionalities integrated that can be looked up in the Dash documentation (like removal of datapoints, changing values etc.)"""
        if clickData is None:
            return existing_table

        selected_feature = clickData['points'][0]['customdata']

        # Rueckpicking: change df['selected'] to True in plot
        identifier_column = df.columns[0]
        row_index = df[df[identifier_column] == selected_feature[0]].index
        # If the row exists, update the 'selected' column to True
        if not row_index.empty:
            df.at[row_index[0], 'selected'] = True

        selected_feature[df.columns.get_loc('selected')] = True  # Rueckpicking: change df['selected'] to True in table

        logging.info(selected_feature)
        print(selected_feature)
            
        # Create a new row for the table
        new_row = dict(zip(df.columns, selected_feature))

        # If the existing_table is None (first time), use an empty table
        if existing_table is None:
            existing_table = []

        # Append the new row to the existing table
        return existing_table + [new_row]


    # Callback to update the plot based on the selected legend attribute
    @app.callback(
        Output('plot', 'figure'),
        [Input('legend-attribute', 'value')]
    )
    def update_plot(legend_attribute):
        fig = px.scatter(
            df,
            x=X_red[:, 0],
            y=X_red[:, 1],
            color=df[legend_attribute],  # Update color by selected attribute
            title=f'2D {method} on dataset {project_name}',
            hover_data=df.columns,
            opacity=0.8,
            symbol=df['marker_symbol'],
            size=df['marker_size'],
            color_continuous_scale=px.colors.sequential.Viridis
        )
        fig.update_layout(margin={'l': 40, 'b': 40, 't': 50, 'r': 0})
        return fig

    return app


 
if __name__ == '__main__':
    import pandas as pd
    
    # Dummy dataset and dimensionality reduced data for testing
    df = pd.DataFrame({
        'uid': [1, 2, 3, 4],
        'cluster': ['A', 'B', 'A', 'C'],
        'taxid': [1, 2, 3, 4],
        'ec': [1, 2, 3, 4],
        'taxid_name': ['abc', 'def', 'ghi', 'abc'],
        'selected': ['False', 'False', 'False', 'False'],
        'marker_symbol': ['circle', 'square', 'diamond', 'triangle-up'],
        'marker_size': [10, 20, 30, 40]
    })
    
    import numpy as np
    X_red = np.random.rand(4, 2)  # Simulate some 2D reduced data
    
    # Initialize Dash app
    app = dash.Dash(__name__)
    
    # Run the app
    run_dash_app(df, X_red, 'PCA', 'Project name', app).run_server(debug=True)