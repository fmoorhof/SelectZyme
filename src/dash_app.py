"""Definitions for the Dash app.

Dash wiki was insanely helpful for the DataTable. If you like to implement additional features, check out the documentation:
https://dash.plotly.com/datatable/editable"""
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

from visualizer import plot_2d, plot_3d


def run_dash_app(df, X_red, method: str, project_name: str):
    """Generate layout and callbacks for a dimensionality reduction page."""
    cols = df.columns.values.tolist()

    fig = plot_2d(df, X_red, legend_attribute=cols[2])

    # Define the layout of the Dash app
    layout = html.Div([
        # Dropdown to select legend attribute of df columns
        html.Div([
            dcc.Dropdown(
                id='legend-attribute',
                options = [{'label': col, 'value': col} for col in cols],  # cols[:12]
                value=cols[2]  # set default column to show on loading
            )
        ], style={'width': '30%', 'display': 'inline-block'}),

        # Scatter plot
        dcc.Graph(
            id='plot',
            figure=fig,        
            config={'scrollZoom': True,},
            style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
        ),

    dash_table.DataTable(
        id='data-table',
        columns=[{'id': c, 'name': c} for c in df.columns],
        style_cell={'textAlign': 'left'},
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
    ])

    # Return layout and callbacks
    def register_callbacks(app):
        # Define callbacks
        @app.callback(
            Output('data-table', 'data'),
            Input('plot', 'clickData'),
            dash.dependencies.State('data-table', 'data')
        )
        def update_table(clickData, existing_table):
            if clickData is None:
                return existing_table

            selected_feature = clickData['points'][0]['customdata']

            identifier_column = df.columns[0]
            row_index = df[df[identifier_column] == selected_feature[0]].index

            if not row_index.empty:
                df.at[row_index[0], 'selected'] = True

            selected_feature[df.columns.get_loc('selected')] = True
            selected_feature[df.columns.get_loc('sequence')] = selected_feature[df.columns.get_loc('sequence')].replace('<br>', '')
                
            new_row = dict(zip(df.columns, selected_feature))

            if existing_table is None:
                existing_table = []

            return existing_table + [new_row]

        @app.callback(
            Output('plot', 'figure'),
            Input('legend-attribute', 'value')
        )
        def update_plot(legend_attribute):
            return plot_2d(df, X_red, legend_attribute)

    return layout, register_callbacks
