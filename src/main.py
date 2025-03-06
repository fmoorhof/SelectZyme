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

from ydata_profiling import ProfileReport

from src.customizations import custom_plotting, set_columns_of_interest
from src.embed import gen_embedding
from src.ml import dimred_caller, perform_hdbscan_clustering
from src.mst_plotting import MinimumSpanningTree
from src.preprocessing import Preprocessing
from src.single_linkage_plotting import create_dendrogram
from src.utils import parse_data
from src.vector_db import QdrantDB
from src.visualizer import plot_2d


def main(config, db_host: str = "http://ocean:6333") -> None:
    # backend calculations
    df = parse_data(
        config["project"]["name"],
        config["project"]["data"]["query_terms"],
        config["project"]["data"]["length"],
        config["project"]["data"]["custom_data_location"],
        config["project"]["data"]["out_dir"],
        config["project"]["data"]["df_coi"],
    )
    logging.info(f"df columns have the dtypes: {df.dtypes}")

    if config["project"]["preprocessing"]:
        df = Preprocessing(df).preprocess()

    if config["project"]["use_DB"]:
        # Load embeddings from Vector DB
        db = QdrantDB(
            collection_name=config["project"]["name"],
            host=db_host
            )
        X = db.database_access(
            df=df, plm_model=config["project"]["plm"]["plm_model"]
        )
    else:
        X = gen_embedding(
            sequences=df["sequence"].tolist(),
            plm_model=config["project"]["plm"]["plm_model"],
        )

    # Clustering
    labels, G, Gsl, X_centroids = perform_hdbscan_clustering(
        X,
        config["project"]["clustering"]["min_samples"],
        config["project"]["clustering"]["min_cluster_size"],
    )
    
    # apply customizations
    df["cluster"] = labels
    df = custom_plotting(df, 
                         config["project"]["plot_customizations"]["size"], 
                         config["project"]["plot_customizations"]["shape"])

    # Dimensionality reduction
    X_red, X_red_centroids = dimred_caller(
        X,
        X_centroids,
        config["project"]["dimred"]["method"],
        config["project"]["dimred"]["n_neighbors"],
        config["project"]["dimred"]["random_state"],
    )

    # Create plots (files are saved under results/)
    fig_dim = plot_2d(df, X_red, X_red_centroids, legend_attribute="cluster")
    mst = MinimumSpanningTree(G._mst, df, X_red, fig_dim)
    fig_mst = mst.plot_mst_in_dimred_landscape()
    fig_slc = create_dendrogram(Z=Gsl._linkage, df=df)

    # EDA (file is saved under assets/)
    columns_of_interest = set_columns_of_interest(df.columns)

    df_profile = df[columns_of_interest]
    df_profile.drop(columns=["accession"], inplace=True)

    profile = ProfileReport(
        df_profile, title="Profiling Report", config_file=""
    )
    try:
        profile.to_file("assets/eda.html")
    except Exception as e:
        logging.error(f"Failed to generate EDA report: {e}")
        with open("assets/eda.html", "w") as f:
            f.write(f"<html><body><h1>EDA Report could not be generated because of: {e}</h1></body></html>")

    return fig_dim, fig_mst, fig_slc  #, fig_eda  # todo: pass variables or simply write files, whats needed for communication to front-end part?



if __name__ == "__main__":
    from src.utils import parse_args

    # CLI argument parsing
    config = parse_args()

    main(config, db_host="http://ocean:6333")
