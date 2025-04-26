"""
This script is the main entry point for the core part of SelectZyme to be used for the SelectZyme web application. It is responsible for the following tasks:

0. Parsing of input data
1. Parsing additional UniProt data (optional)
2. Preprocessing the data
3. Generating embeddings
4. Clustering the embeddings
5. Dimensionality reduction of the embeddings
6. Customizing the plot
7. Saving the plots to be intergrated into the front end of the web application
"""
from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO)

import copy

from ydata_profiling import ProfileReport

from selectzyme.backend.customizations import custom_plotting, set_columns_of_interest
from selectzyme.backend.embed import gen_embedding
from selectzyme.backend.ml import dimred_caller, perform_hdbscan_clustering
from selectzyme.backend.preprocessing import Preprocessing
from selectzyme.backend.utils import parse_data
from selectzyme.backend.vector_db import QdrantDB
from selectzyme.frontend.mst_plotting import MinimumSpanningTree
from selectzyme.frontend.single_linkage_plotting import create_dendrogram
from selectzyme.frontend.visualizer import plot_2d


def load_and_preprocess(config):
    # Parse data and preprocess if needed
    df = parse_data(
        config["project"]["name"],
        config["project"]["data"]["query_terms"],
        config["project"]["data"]["length"],
        config["project"]["data"]["custom_data_location"],
        config["project"]["data"]["out_dir"],
        config["project"]["data"]["df_coi"],
    )
    if config["project"]["preprocessing"]:
        df = Preprocessing(df).preprocess()
    logging.info(f"DataFrame dtypes: {df.dtypes}")

    df = custom_plotting(
        df, 
        config["project"]["plot_customizations"]["size"],
        config["project"]["plot_customizations"]["shape"]
    )
    return df


def generate_embeddings(config, df, db_host):
    # Get embeddings based on config
    if config["project"]["use_DB"]:
        db = QdrantDB(collection_name=config["project"]["name"], host=db_host)
        X = db.database_access(df=df, plm_model=config["project"]["plm"]["plm_model"])
    else:
        X = gen_embedding(
            sequences=df["sequence"].tolist(),
            plm_model=config["project"]["plm"]["plm_model"],
        )
    return X


def perform_clustering(config, X, df):
    # Perform clustering and return labels and associated outputs.
    return perform_hdbscan_clustering(
        X,
        df,
        config["project"]["clustering"]["min_samples"],
        config["project"]["clustering"]["min_cluster_size"],
    )


def create_visualizations(df, X_red, G, Gsl):
    fig_dim = plot_2d(df, X_red, legend_attribute="cluster")
    fig_mst = copy.deepcopy(fig_dim)  # deep copy needed else fig_dim will be modified by next line
    fig_mst = MinimumSpanningTree(G._mst, df, X_red, fig_mst).plot_mst_in_dimred_landscape()
    fig_slc = create_dendrogram(Z=Gsl._linkage, df=df)
    return fig_dim, fig_mst, fig_slc


def main(config, db_host: str = "http://ocean:6333") -> None:
    export_path = config["project"]["data"]["out_dir"] + config["project"]["name"]

    df = load_and_preprocess(config)

    X = generate_embeddings(config, df, db_host)
    G, Gsl, df = perform_clustering(config, X, df)

    # Dimensionality reduction
    X_red = dimred_caller(
        X,
        config["project"]["dimred"]["method"],
        config["project"]["dimred"]["n_neighbors"],
        config["project"]["dimred"]["random_state"],
    )

    fig_dim, fig_mst, fig_slc = create_visualizations(df, X_red, G, Gsl)
    fig_dim.write_html(export_path + "_dimred.html")
    fig_mst.write_html(export_path + "_mst.html")
    fig_slc.write_html(export_path + "_slc.html")

    # Generate EDA Report
    columns_of_interest = set_columns_of_interest(df.columns)
    df_profile = df[columns_of_interest].drop(columns=["accession"])
    profile = ProfileReport(df_profile, title="Profiling Report", config_file="")
    try:
        profile.to_file("assets/eda.html")
        profile.to_file(export_path + "_eda.html")
    except Exception as e:
        logging.error(f"Failed to generate EDA report: {e}")
        with open("assets/eda.html", "w") as f:
            f.write(f"<html><body><h1>EDA Report could not be generated because: {e}</h1></body></html>")
    return fig_dim, fig_mst, fig_slc, profile



if __name__ == "__main__":
    from backend.utils import parse_args

    # CLI argument parsing
    config = parse_args()

    main(config, db_host="http://ocean:6333")
