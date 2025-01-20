from base64 import b64encode
import io

import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.figure_factory as ff

from src.hdbscan_plotting import SingleLinkageTree


def layout(G, df: pd.DataFrame, polar=False) -> html.Div:
    sl = SingleLinkageTree(G._linkage, df)
    fig = sl.plot(polar=polar)  # latest implementation using the hdbscan_plotting
    
    # attempt with the plotly figure factory: dendrogram front-end rendering ultra slow. browser freezes!
    # but implementation correct and very pretty, maybe check why so slow and optimize
    fig = ff.create_dendrogram(G._linkage)  # todo: labels=df


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
        # todo: why customdata not working yet?
        # accession  = clickData['points'][0]['customdata']  # accession  = clickData['points'][0]['text'].split('<br>')[0].replace('accession: ', '')  # if customdata fails 
        accession  = clickData['points'][0]['text'].split('<br>')[0].replace('accession: ', '')
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