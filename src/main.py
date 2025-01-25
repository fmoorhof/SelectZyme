"""This file soon can be deleted together with dash_app.py but before merge some functions required by app.py
into there or a utils file."""
import logging
import sys
import os
import argparse

import yaml
import pandas as pd
from qdrant_client import QdrantClient
import dash
import numpy as np

from preprocessing import Parsing
from preprocessing import Preprocessing
from embed import load_or_createDB
import visualizer
from customizations import custom_plotting
from fetch_data_uniprot import UniProtFetcher
from dash_app import run_dash_app


logging.basicConfig(
    format="%(levelname)-8s| %(module)s.%(funcName)s: %(message)s", level=logging.DEBUG
)


def parse_args() -> argparse.Namespace:
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
                        help='Dimensionality reduction technique (default: PCA)')
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


def preprocessing(df: pd.DataFrame):
    # df needs to contain a column 'sequence' with the sequences
    pp = Preprocessing(df)
    pp.remove_long_sequences()
    pp.remove_sequences_without_Metheonin()
    pp.remove_sequences_with_undertermined_amino_acids()
    pp.remove_duplicate_entries()
    pp.remove_duplicate_sequences()
    return pp.df


def parse_data(args):
    """Parse data or read it from file
    todo: implement .xlsx reader with spreadsheet name"""
    input_file = args.out_dir+args.project_name+'_annotated.tsv'
    if not os.path.isfile(input_file):  # generate it
        fetcher = UniProtFetcher(args.df_coi, args.out_dir)
        df = fetcher.query_uniprot(args.query_terms, args.length)
        if args.custom_data_location != '':
            if args.custom_data_location.endswith('.fasta'):
                custom_data = fetcher.load_custom_fasta(args.custom_data_location)
            else:
                custom_data = fetcher.load_custom_csv(file_path=args.custom_data_location, sep=';')
            df = pd.concat([custom_data, df], ignore_index=True)
        df = fetcher.clean_data(df)
        fetcher.save_data(df, args.project_name)
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
    qdrant = QdrantClient(path="/data/tmp/EnzyNavi")  # path= write them to disk OR use memory instance ":memory:"  # perf: instantiation very slow
    annotation, embeddings = load_or_createDB(qdrant, df, collection_name=project_name, plm_model=plm_model)
    if df.shape[0] != embeddings.shape[0]:  # todo: recreate the collection instead and make user aware (press [Yn] or dont know yet to report)
        qdrant.delete_collection(collection_name=project_name)  # delete a collection because it is supposed to have changed in the meantime
        raise ValueError(f"Length of dataframe ({df.shape[0]}) and embeddings ({embeddings.shape[0]}) do not match. As a consequence, the collection is deleted and you need to embed again. So just re-run.")

    sys.setrecursionlimit(max(df.shape[0], 10000))  # fixed: RecursionError: maximum recursion depth exceeded

    return embeddings


def dimred_clust(df, X, dim_method):
    labels, G, Gsl, X_centroids = visualizer.clustering_HDBSCAN(X, min_samples=3, min_cluster_size=10)  # min samples for batch7: 50  # perf: the higher the parameters, the quicker HDBSCAN runs
    df['cluster'] = labels
    df = custom_plotting(df)

    dim_method = dim_method.upper()
    if dim_method == 'PCA':
        X_red, X_red_centroids = visualizer.pca(X, X_centroids)
    elif dim_method == 'TSNE':
        X_red = visualizer.tsne(X, random_state=42)
        X_red_centroids = np.empty((0, 2))
    elif dim_method == 'UMAP':
        X_red, X_red_centroids = visualizer.umap(X, X_centroids, n_neighbors=15, random_state=42)

    return df, X_red, G, Gsl, X_red_centroids


def main(app):
    df = parse_data(args)
    df = preprocessing(df)

    X = database_access(df, args.project_name)
    df, X_red, G, Gsl = dimred_clust(df, X, args.dim_red)

    # TSNE plot and table app
    # app = run_dash_app(df, X_red, args.dim_red, args.project_name, app)  # plot and table: from dash_app import run_dash_app
    app.layout, register_callbacks = run_dash_app(df, X_red)  # plot and table: from dash_app import run_dash_app
    register_callbacks(app)



if __name__ == "__main__":
    app = dash.Dash(__name__)
    args = argparse.Namespace(project_name='argparse_test', query_terms=["ec:1.13.11.85", "latex clearing protein"], length='200 TO 601', custom_data_location="/raid/data/fmoorhof/PhD/SideShit/LCP/custom_seqs_no_signals.csv", dim_red='TSNE', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    # args = argparse.Namespace(project_name='argparse_test_minimal', query_terms=["ec:1.13.11.85", "ec:1.13.11.84"], length='200 TO 601', custom_data_location="/raid/data/fmoorhof/PhD/SideShit/LCP/custom_seqs_no_signals.csv", dim_red='PCA', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    args = parse_args()  # comment for debugging

    # our manual mining
    # args = argparse.Namespace(project_name='batch7', query_terms=["ec:1.14.11", "ec:1.14.20","xref%3Abrenda-1.14.11", "xref%3Abrenda-1.14.20", "IPR005123", "IPR003819", "IPR026992", "PF03171", "2OG-FeII_Oxy", "cd00250"], length='201 TO 500', custom_data_location="/raid/data/fmoorhof/PhD/Data/SKD022_2nd-order/custom_seqs_full.csv", dim_red='TSNE', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    # args = argparse.Namespace(project_name='test', query_terms=["xref%3Abrenda-1.14.20", "xref%3Abrenda-1.14.11"], length='201 TO 500', custom_data_location="/raid/data/fmoorhof/PhD/Data/SKD022_2nd-order/expoImpoTest.csv", dim_red='TSNE', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    # query_terms = ["ec:1.13.11.85", "xref%3Abrenda-1.13.11.85", "ec:1.13.11.87", "xref%3Abrenda-1.13.11.87", "ec:1.13.99.B1", "xref%3Abrenda-1.13.99.B1", "IPR037473", "IPR018713", "latex clearing protein"]  # define your query terms for UniProt here
    # args = argparse.Namespace(project_name='petase', query_terms=query_terms, length='50 TO 1020', custom_data_location='/raid/data/fmoorhof/PhD/Data/SKD021_Case_studies/PETase/pet_plasticDB_preprocessed.csv', dim_red='PCA', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])

    main(app=app)
    app.run_server(host='0.0.0.0', port=8050, debug=False)  # debug=True triggers main() execution twice
    
    # old project names to remember: batch5, test_project, lcp, lcp_no_signals, lefos, lefos_no_signals
    # main(input_file='/raid/data/fmoorhof/PhD/Data/SKD001_Literature_Mining/Batch5/batch5_annotated.tsv', project_name='batch5', app=app)    
    # from docker (no matter is docker or not) to local machine: http://192.168.3.156:8050/
    # http://10.10.142.201:8050/
