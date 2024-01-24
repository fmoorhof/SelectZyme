import logging
import sys

import pandas as pd
from qdrant_client import QdrantClient, models

from preprocessing import Parsing
from preprocessing import Preprocessing
import embed
import visualizer


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
    X_red = visualizer.pca(X)
    visualizer.plot_2d(df, X_red, collection_name=project_name, method='PCA')



if __name__ == "__main__":
    main(input_file='tests/head_10.tsv', project_name='test_project')
