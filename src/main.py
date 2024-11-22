import logging
import sys
import os
import argparse

import pandas as pd
from qdrant_client import QdrantClient
import dash

from preprocessing import Parsing
from preprocessing import Preprocessing
from embed import load_or_createDB
import visualizer
from dash_app import run_dash_app
from fetch_data_uniprot import UniProtFetcher


logging.basicConfig(
    format="%(levelname)-8s| %(module)s.%(funcName)s: %(message)s", level=logging.DEBUG
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Process parameters.')
    parser.add_argument('-p', '--project_name', required=True, help='Project name')
    parser.add_argument('-q', '--query_terms', nargs='+', required=True, help='Query terms for UniProt')
    parser.add_argument('--length', required=True, help='Length range for sequences')
    parser.add_argument('-loc', '--custom_data_location', required=True, help='Location of your custom data CSV')
    parser.add_argument('-o', '--out_filename', required=True, help='Output filename')

    # Optional arguments with defaults
    parser.add_argument('--dim_red', default='TSNE', 
                        help='Dimensionality reduction technique (default: TSNE)')
    parser.add_argument('--out_dir', default='datasets/output/', 
                        help='Output directory (default: datasets/output/)')
    parser.add_argument('--df_coi', nargs='+', default=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'], 
                        help=('Define the columns of interest for the dataframe '
                              '(default: accession, reviewed, ec, organism_id, length, xref_brenda, xref_pdb, sequence)'))

    args = parser.parse_args()

    logging.debug(f"Project name: {args.project_name}")
    logging.debug(f"Query terms: {args.query_terms}")
    logging.debug(f"Length range: {args.length}")
    logging.debug(f"Custom data location: {args.custom_data_location}")
    logging.debug(f"Output filename: {args.out_filename}")
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
    """Parse data or read it from file"""
    input_file = args.out_dir+args.out_filename+'_annotated.tsv'
    if not os.path.isfile(input_file):  # generate it
        fetcher = UniProtFetcher(args.df_coi, args.out_dir)
        df = fetcher.query_uniprot(args.query_terms, args.length)
        custom_data = fetcher.load_custom_csv(args.custom_data_location)
        df = pd.concat([custom_data, df], ignore_index=True)
        df = fetcher.clean_data(df)
        fetcher.save_data(df, args.out_filename)
    elif input_file.endswith('.fasta'):
        headers, sequences = Parsing.parse_fasta(input_file)
        df = pd.DataFrame({'Header': headers, 'Sequence': sequences})
    else:
        df = Parsing.parse_tsv(input_file)
    return df


def database_access(df, project_name):
    # Create a collection in Qdrant DB with embedded sequences
    qdrant = QdrantClient(path="/data/tmp/EnzyNavi")  # OR write them to disk
    annotation, embeddings = load_or_createDB(qdrant, df, collection_name=project_name)
    if df.shape[0] != embeddings.shape[0]:
        qdrant.delete_collection(collection_name=project_name)  # delete a collection because it is supposed to have changed in the meantime
        raise ValueError(f"Length of dataframe ({df.shape[0]}) and embeddings ({embeddings.shape[0]}) do not match. As a consequence, the collection is deleted and you need to embed again. So just re-run.")

    sys.setrecursionlimit(max(df.shape[0], 10000))  # fixed: RecursionError: maximum recursion depth exceeded

    return embeddings


def dimred_clust(df, X, dim_method):
    labels = visualizer.clustering_HDBSCAN(X, min_samples=1, min_cluster_size=5)  # min samples for batches: 50
    df['cluster'] = labels
    df = visualizer.custom_plotting(df)

    dim_method = dim_method.upper()
    if dim_method == 'PCA':
        X_red = visualizer.pca(X)
    elif dim_method == 'TSNE':
        X_red = visualizer.tsne(X, random_state=42)
    elif dim_method == 'UMAP':
        X_red = visualizer.umap(X, n_neighbors=15, random_state=42)

    return df, X_red


def main(app):
    df = parse_data(args)
    df = preprocessing(df)

    X = database_access(df, args.project_name)
    df, X_red = dimred_clust(df, X, args.dim_red)

    app = run_dash_app(df, X_red, args.dim_red, args.project_name, app)



if __name__ == "__main__":
    app = dash.Dash(__name__)
    args = argparse.Namespace(project_name='argparse_test', query_terms=["ec:1.13.11.85", "latex clearing protein"], length='200 TO 601', custom_data_location="/raid/data/fmoorhof/PhD/SideShit/LCP/custom_seqs_no_signals.csv", out_filename='argparse_test', dim_red='TSNE', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    # args = parse_args()  # comment for debugging

    main(app=app)
    app.run_server(host='0.0.0.0', port=8050, debug=False)  # debug=True triggers main() execution twice
    
    # old project names to remember: batch5, test_project, lcp, lcp_no_signals, lefos, lefos_no_signals
    # main(input_file='/raid/data/fmoorhof/PhD/Data/SKD001_Literature_Mining/Batch5/batch5_annotated.tsv', project_name='batch5', app=app)    
    # from docker (no matter is docker or not) to local machine: http://192.168.3.156:8050/
    # http://10.10.142.201:8050/
