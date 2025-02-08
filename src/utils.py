import logging
import sys
import os
import argparse

import yaml
import pandas as pd
from qdrant_client import QdrantClient

from src.fetch_data_uniprot import UniProtFetcher
from src.parsing import Parsing
from src.vector_db import load_or_createDB
from src import ml
from src.customizations import custom_plotting



def parse_args():
    parser = argparse.ArgumentParser(description='Process parameters.')

    parser.add_argument('--config', type=str, default='results/test_config.yml', help='Path to a config.yml file with all parameters (default location: results/test_config.yml)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required arguments
    if not all([config['project']['data'], config['project']['plm']]):
        parser.error("Some required arguments, either CLI or from config.yml, not provided.")

    return config


def parse_data(project_name, query_terms, length, custom_file, out_dir, df_coi):
    exisiting_file = out_dir+project_name+'.tsv'
    output_file_corpus = out_dir+project_name
    df = _parse_data(exisiting_file, custom_file, query_terms, length, df_coi)
    df = _clean_data(df)
    _save_data(df, output_file_corpus)
    return df


def database_access(df: pd.DataFrame, project_name: str, plm_model: str = 'esm1b'):
    """Create a collection in Qdrant DB with embedded sequences
    :param df: dataframe containing the sequences and the annotation
    :param project_name: name of the collection
    return: embeddings: numpy array containing the embeddings"""
    logging.info("Instantiating Qdrant vector DB. This takes quite a while.")
    qdrant = QdrantClient(url="http://localhost:6333", timeout=15)  # fire up container with  # docker run -p 6333:6333 -p 6334:6334 -v "/data/tmp/EnzyNavi/qdrant_storage:/qdrant/storage:z" fmoorhof/qdrant:1.13.2
    annotation, embeddings = load_or_createDB(qdrant, df, collection_name=project_name, plm_model=plm_model)
    if df.shape[0] != embeddings.shape[0]:  # todo: recreate the collection instead and make user aware (press [Yn] or dont know yet to report)
        qdrant.delete_collection(collection_name=project_name)  # delete a collection because it is supposed to have changed in the meantime
        raise ValueError(f"Length of dataframe ({df.shape[0]}) and embeddings ({embeddings.shape[0]}) do not match. As a consequence, the collection is deleted and you need to embed again. So just re-run.")

    sys.setrecursionlimit(max(df.shape[0], 10000))  # fixed: RecursionError: maximum recursion depth exceeded

    return embeddings


def dimred_clust(df, X, dim_method, n_neighbors=15, random_state=42):
    # todo: adapt argparsing for HDBSCAN to
    labels, G, Gsl, X_centroids = ml.clustering_HDBSCAN(X, min_samples=5, min_cluster_size=10)  # min samples for batch7: 50  # perf: the higher the parameters, the quicker HDBSCAN runs
    df['cluster'] = labels
    df = custom_plotting(df)

    dim_method = dim_method.upper()
    if dim_method == 'PCA':
        X_red, X_red_centroids = ml.pca(X, X_centroids)
    elif dim_method == 'TSNE':
        X_red, X_red_centroids = ml.tsne(X, random_state=random_state)
    elif dim_method == 'OPENTSNE':
        X_red, X_red_centroids = ml.opentsne(X, X_centroids, random_state=random_state)
    elif dim_method == 'UMAP':
        X_red, X_red_centroids = ml.umap(X, X_centroids, n_neighbors=n_neighbors, random_state=random_state)
    else:
        raise ValueError(f"Dimensionality reduction method {dim_method} not implemented. Choose from 'PCA', 'TSNE', 'openTSNE', 'UMAP'.")
    
    return df, X_red, G, Gsl, X_red_centroids


def _parse_data(exisiting_file: str, custom_file: str, query_terms: list, length: str, df_coi: list):
    if os.path.isfile(exisiting_file):
        return Parsing(exisiting_file).parse()
    
    if query_terms != ['']:
        fetcher = UniProtFetcher(df_coi)
    if custom_file != '':
        df_custom = Parsing(custom_file).parse()
        if query_terms == ['']:
            return df_custom
        df = fetcher.query_uniprot(query_terms, length)
        df = pd.concat([df_custom, df], ignore_index=True)  # custom data first that they are displayed first in plot legends
        return df
    elif query_terms != ['']:
        return fetcher.query_uniprot(query_terms, length)
    else:
        raise ValueError("No query terms or custom data location provided. Please provide either one.")


def _save_data(df: pd.DataFrame, out_file: str):
    df.to_csv(f"{out_file}.tsv", sep='\t', index=False)
    # _export_annotated_fasta(df, f"{out_file}.fasta")
    # df = df[(df['reviewed'] == True) | (df['xref_brenda'].notnull())]  # | = OR
    # df.to_csv(f"{out_file}.tsv", sep='\t', index=False)


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['accession'] != 'Entry']  # remove concatenated headers that are introduced by each query term
    logging.info(f'Total amount of retrieved entries: {df.shape[0]}')
    df.drop_duplicates(subset='accession', keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    logging.info(f'Total amount of non redundant entries: {df.shape[0]}')
    logging.info(f"Amount of BRENDA reviewed entries: {df['xref_brenda'].notna().sum()}")
    return df


def _export_annotated_fasta(df: pd.DataFrame, out_file: str):
    with open(out_file, 'w') as f_out:
        for index, row in df.iterrows():
            fasta = '>', '|'.join(row.iloc[:-1].map(str)), '\n', row.loc["sequence"], '\n'  # todo: exclude seq in header (-1=last column potentially broken)
            f_out.writelines(fasta)
    logging.info(f"FASTA file written to {out_file}")