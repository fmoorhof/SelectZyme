import logging
import sys

import pandas as pd
from qdrant_client import QdrantClient
import dash

from preprocessing import Parsing
from preprocessing import Preprocessing
from embed import load_or_createDB
import visualizer
from dash_app import run_dash_app

logging.basicConfig(
    format="%(levelname)-8s| %(module)s.%(funcName)s: %(message)s", level=logging.DEBUG
)


def preprocessing(df: pd.DataFrame):
    # df needs to contain a column 'Sequence' with the sequences
    pp = Preprocessing(df)
    pp.remove_long_sequences()
    pp.remove_sequences_without_Metheonin()
    pp.remove_sequences_with_undertermined_amino_acids()
    pp.remove_duplicate_entries()
    pp.remove_duplicate_sequences()
    return pp.df


def main(input_file: str, project_name: str, app):

    # data parsing
    if input_file.endswith('.fasta'):
        headers, sequences = Parsing.parse_fasta(input_file)
        df = pd.DataFrame({'Header': headers, 'Sequence': sequences})
    else:
        df = Parsing.parse_tsv(input_file)

    df = preprocessing(df)

    # Create a collection in Qdrant DB with embedded sequences
    qdrant = QdrantClient(path="/scratch/global_1/fmoorhof/Databases/Vector_db/")  # OR write them to disk
    annotation, embeddings = load_or_createDB(qdrant, df, collection_name=project_name)
    if df.shape[0] != embeddings.shape[0]:
        qdrant.delete_collection(collection_name=project_name)  # delete a collection because it is supposed to have changed in the meantime
        raise ValueError(f"Length of dataframe ({df.shape[0]}) and embeddings ({embeddings.shape[0]}) do not match. As a consequence, the collection is deleted and you need to embed again. So just re-run.")

    sys.setrecursionlimit(max(df.shape[0], 10000))  # fixed: RecursionError: maximum recursion depth exceeded
    X = embeddings
    labels = visualizer.clustering_HDBSCAN(X, min_samples=1, min_cluster_size=250)  # min samples for batches: 50
    df['cluster'] = labels
    df = visualizer.custom_plotting(df)

    iter_methods = ['TSNE']  # ['PCA', 'TSNE', 'UMAP']
    for method in iter_methods:
        if method == 'PCA':
            X_red = visualizer.pca(X)
        elif method == 'TSNE':
            X_red = visualizer.tsne(X, random_state=42)
        elif method == 'UMAP':
            X_red = visualizer.umap(X, n_neighbors=15, random_state=42)
        visualizer.plot_2d(df, X_red, collection_name=project_name, method=method)

    app = run_dash_app(df, X_red, method, project_name, app)



if __name__ == "__main__":
    app = dash.Dash(__name__)
    main(input_file='tests/head_10.tsv', project_name='test_project', app=app)
    # main(input_file='datasets/output/uniprot_lcp_annotated.tsv', project_name='lcp', app=app)  # uniprot_lcp_annotated
    # main(input_file='/raid/data/fmoorhof/PhD/Data/SKD001_Literature_Mining/Batch5/batch5_annotated.tsv', project_name='batch5', app=app)
    app.run_server(host='0.0.0.0', port=8050, debug=False)  # debug=True triggers main() execution twice
    # from docker (no matter is docker or not) to local machine: http://192.168.3.156:8050/
    # http://10.10.142.201:8050/