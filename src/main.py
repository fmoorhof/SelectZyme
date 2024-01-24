import logging
import sys

import pandas as pd
from qdrant_client import QdrantClient, models

from preprocessing import Parsing
from preprocessing import Preprocessing
import embed
import visualizer

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
app = dash.Dash(__name__)

logging.basicConfig(
    format="%(levelname)-8s| %(module)s.%(funcName)s: %(message)s", level=logging.DEBUG
)


def db_creation(df, collection_name: str):
    qdrant = QdrantClient(path="datasets/Vector_db/")  # OR write them to disk
    collections_info = qdrant.get_collections()
    if collection_name not in str(collections_info):  # todo: implement this nicely: access the 'name' field of the object
        embeddings = embed.gen_embedding(df['Sequence'].tolist(), device='cuda')
        annotation = embed.create_vector_db_collection(qdrant, df, embeddings, collection_name=collection_name)
    else:
        annotation, embeddings = embed.load_collection_from_vector_db(qdrant, collection_name)
    return annotation, embeddings


def main(input_file: str, project_name: str):

    if input_file.endswith('.fasta'):
        headers, sequences = Parsing.parse_fasta(input_file)
        df = pd.DataFrame({'Header': headers, 'Sequence': sequences})
    else:
        df = Parsing.parse_tsv(input_file)

    # df needs to contain a column 'Sequence' with the sequences
    pp = Preprocessing(df)
    pp.remove_long_sequences()
    pp.remove_sequences_without_Metheonin()
    pp.remove_sequences_with_undertermined_amino_acids()
    pp.remove_duplicate_entries()
    pp.remove_duplicate_sequences()
    df = pp.df


    # Create a collection in Qdrant DB with embedded sequences
    annotation, embeddings = db_creation(df, collection_name=project_name)
    if df.shape[0] != embeddings.shape[0]:
        raise ValueError(f"Length of dataframe ({df.shape[0]}) and embeddings ({embeddings.shape[0]}) do not match. Something went wrong.")

    sys.setrecursionlimit(max(df.shape[0], 10000))  # fixed: RecursionError: maximum recursion depth exceeded
    X = embeddings
    labels = visualizer.clustering_HDBSCAN(X, min_samples=1)
    df = visualizer.custom_plotting(df, labels)

    iter_methods = ['PCA', 'TSNE', 'UMAP']
    for method in iter_methods:
        if method == 'PCA':
            X_red = visualizer.pca(X)
        elif method == 'TSNE':
            X_red = visualizer.tsne(X)
        elif method == 'UMAP':
            X_red = visualizer.umap(X)
        visualizer.plot_2d(df, X_red, collection_name=project_name, method=method)



    # todo: outsource this to a separate file or visualizer.py

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
        html.Table(id='data-table')
    ])

    empty_table = []

    @app.callback(
        Output('data-table', 'children'),
        Input('plot', 'clickData'),
        [dash.dependencies.State('data-table', 'children')]
    )

    def update_table(clickData, existing_table):
        if clickData is None:
            return existing_table
    
        selected_feature = clickData['points'][0]['customdata']
        logging.info(selected_feature)
        print(selected_feature)
    
        # Create a new row for the table
        new_row = [
            html.Tr([
                html.Td(','.join(map(str, selected_feature))),  # show only specific values: [selected_feature[i] for i in [1, 2, 3, 6]]
                html.Td(html.A("BRENDA link", href=selected_feature[-1], target="_blank"))  # selected_feature[-1] = brenda url
            ]),  # join: print with delimiter
        ]
    
        # If the existing_table is None (first time), use an empty table
        if existing_table is None:
            existing_table = empty_table
    
        # Append the new row to the existing table
        return existing_table + new_row
    
    # app.run_server(host='192.168.3.156', port=8050, debug=False)  # from ocean to local machine
    print('done')



if __name__ == "__main__":
    main(input_file='tests/head_10.tsv', project_name='test_project')
    app.run_server(host='0.0.0.0', port=8051, debug=True)  # from docker (no matter is docker or not) to local machine: http://192.168.3.156:8051/

    # time python src/main.py && time python src/main.py
    # main(input_file='/raid/data/fmoorhof/PhD/Data/SKD001_Literature_Mining/Batch5/batch5_annotated.tsv', project_name='test_project')
