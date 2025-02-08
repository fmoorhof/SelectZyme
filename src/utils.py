import logging
import sys
import os
import argparse

import yaml
import pandas as pd
from qdrant_client import QdrantClient

from src.fetch_data_uniprot import UniProtFetcher
from src.parsing import Parsing
from src.preprocessing import Preprocessing
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


def preprocessing(df: pd.DataFrame):
    # df needs to contain a column 'sequence' with the sequences
    pp = Preprocessing(df)
    pp.remove_long_sequences()
    pp.remove_sequences_without_Metheonin()
    pp.remove_sequences_with_undertermined_amino_acids()
    pp.remove_duplicate_entries()
    pp.remove_duplicate_sequences()
    return pp.df


def parse_data(project_name, query_terms, length, custom_data_location, out_dir, df_coi):
    """Parse data or read it from file
    todo: implement .xlsx reader with spreadsheet name"""
    input_file = out_dir+project_name+'_annotated.tsv'
    if not os.path.isfile(input_file):  # generate it
        fetcher = UniProtFetcher(df_coi, out_dir)
        df = fetcher.query_uniprot(query_terms, length)
        if custom_data_location != '':
            if custom_data_location.endswith('.fasta'):
                custom_data = fetcher.load_custom_fasta(custom_data_location)
            else:
                custom_data = fetcher.load_custom_csv(file_path=custom_data_location, sep=';')
            df = pd.concat([custom_data, df], ignore_index=True)
        df = fetcher.clean_data(df)
        fetcher.save_data(df, project_name)
    elif input_file.endswith('.fasta'):
        headers, sequences = Parsing.parse_fasta(input_file)
        df = pd.DataFrame({'Header': headers, 'Sequence': sequences})
    else:
        df = Parsing.parse_tsv(input_file)
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


@DeprecationWarning
def parse_args_old() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Process parameters.')
    
    # Add --config argument for YAML file
    parser.add_argument('--config', type=str, help='Path to config.yml file')
    
    # Required arguments (if no config file is provided)
    parser.add_argument('-p', '--project_name', help='Project name')
    parser.add_argument('-q', '--query_terms', nargs='+', help='Query terms for UniProt')
    parser.add_argument('-l', '--length', help='Length range for sequences to retrieve from UniProt')

    # Optional arguments with defaults
    parser.add_argument('-loc', '--custom_data_location', default='', help='Location of your custom data CSV')
    parser.add_argument('--dim_red', default='PCA', 
                        help='Dimensionality reduction technique (default: PCA), other methods: TSNE, openTSNE, UMAP')
    parser.add_argument('--plm_model', default='esm1b', 
                        help="Protein language model (default: 'esm1b') other models: 'esm2', 'esm3', 'prott5', 'prostt5'")
    parser.add_argument('--out_dir', default='datasets/output/', 
                        help='Output directory (default: datasets/output/)')
    parser.add_argument('--df_coi', nargs='+', default=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'], 
                        help=('Define the columns of interest for the dataframe '
                              '(default: accession, reviewed, ec, organism_id, length, xref_brenda, xref_pdb, sequence)'))

    # Parse CLI arguments
    args = parser.parse_args()

    # If a config file is provided, load it and override CLI arguments
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override CLI arguments with values from the config file
        for key, value in config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)

    # Validate required arguments
    if not args.config and not all([args.project_name, args.query_terms, args.length, args.custom_data_location]):
        parser.error("Either a config file or all required arguments (--project_name, --query_terms, --length, --custom_data_location) must be provided.")

    # Log the final arguments
    logging.debug(f"Project name: {args.project_name}")
    logging.debug(f"Query terms: {args.query_terms}")
    logging.debug(f"Length range: {args.length}")
    logging.debug(f"Custom data location: {args.custom_data_location}")
    logging.debug(f"Dimensionality reduction: {args.dim_red}")
    logging.debug(f"Protein Language Model: {args.plm_model}")        
    logging.debug(f"Output directory: {args.out_dir}")
    logging.debug(f"Dataframe columns of interest: {args.df_coi}")

    return args
