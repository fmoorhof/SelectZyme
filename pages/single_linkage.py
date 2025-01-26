from dash import html, dcc, dash_table
import pandas as pd
# import plotly.figure_factory as ff

from pages.dimred import html_export_figure
from src.customizations import set_columns_of_interest
# from src.hdbscan_plotting import SingleLinkageTree
from single_linkage_plotting import create_dendrogram


def layout(G, df: pd.DataFrame) -> html.Div:
    # tree looks a bit creapy after overengineering it. try to revert and merge new functionalities instead of reverting here much (build from scratch again in single_linkage_plotting.py)
    # from src.hdbscan_plotting import SingleLinkageTree
    # sl = SingleLinkageTree(G._linkage, df)
    # fig = sl.plot()

    # attempt with the plotly figure factory: dendrogram front-end rendering ultra slow. also dendrogram creation and figure creation
    # but implementation correct and very pretty, maybe check why so slow and optimize
    columns_of_interest = set_columns_of_interest(df.columns)
    hover_text = ["<br>".join(f"{col}: {df[col][i]}" for col in columns_of_interest) for i in range(len(df))]
    # fig = ff.create_dendrogram(G._linkage, hovertext=hover_text)  # labels=df['accession'].to_list()  # root node causes probably different dimensions of G._linkage and df
    
    # leanest SL implementation and current go-to. Callback also working
    fig = create_dendrogram(Z=G._linkage, df=df, hovertext=hover_text)


    layout = html.Div([
        # plot download button
        html.Div(
            html.A(
            html.Button("Download plot as HTML"), 
            id="download-button",
            href=html_export_figure(fig),  # if other column got selected see callback (update_plot_and_download) for export definition
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
