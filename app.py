from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO)

import os

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import dcc, html
from plotly.graph_objects import Figure

import selectzyme.pages.dimred as dimred
import selectzyme.pages.eda as eda
import selectzyme.pages.mst as mst
import selectzyme.pages.single_linkage as sl
import selectzyme.pages.slc_centroid as sl_centroid
from selectzyme.pages.callbacks import register_callbacks
from selectzyme.backend.customizations import custom_plotting
from selectzyme.backend.embed import gen_embedding
from selectzyme.backend.ml import dimred_caller, perform_hdbscan_clustering
from selectzyme.backend.preprocessing import Preprocessing
from selectzyme.backend.utils import export_annotated_fasta, parse_data
from selectzyme.backend.vector_db import QdrantDB
from selectzyme.frontend.visualizer import plot_2d


def load_and_preprocess(config):
    """
    Load and preprocess data based on the provided configuration.
    This function parses data using the configuration parameters, applies 
    preprocessing if specified, and customizes the data for plotting.
    Args:
        config (dict): A dictionary containing configuration parameters. 
            Expected keys include:
                - "project": A dictionary with the following keys:
                    - "name" (str): The name of the project.
                    - "data": A dictionary with keys:
                        - "query_terms" (list): Terms to query the data.
                        - "length" (int): Length of the data to process.
                        - "custom_data_location" (str): Path to custom data.
                        - "out_dir" (str): Directory to save output data.
                        - "df_coi" (str): DataFrame column of interest.
                    - "preprocessing" (bool): Whether to apply preprocessing.
                    - "plot_customizations": A dictionary with keys:
                        - "size" (int): Size parameter for plotting.
                        - "shape" (str): Shape parameter for plotting.
    Returns:
        pandas.DataFrame: The processed and customized DataFrame.
    """
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

    # apply customizations
    df = custom_plotting(df, 
                         config["project"]["plot_customizations"]["size"], 
                         config["project"]["plot_customizations"]["shape"])
    return df


def load_embeddings(config, df):
    """
    Load embeddings for a given dataset based on the provided configuration.
    This function either retrieves embeddings from a Vector Database (QdrantDB) 
    or generates embeddings using a pre-trained language model (PLM), depending 
    on the configuration.
    Args:
        config (dict): A dictionary containing configuration settings. 
            Expected keys:
                - "project": A dictionary with the following keys:
                    - "use_DB" (bool): If True, embeddings are loaded from the database.
                    - "name" (str): The name of the project/collection in the database.
                    - "plm" (dict): A dictionary with the key:
                        - "plm_model" (str): The name of the pre-trained language model.
        df (pandas.DataFrame): A DataFrame containing the data. It must include 
            a column named "sequence" if embeddings are to be generated.
    Returns:
        numpy.ndarray: An array of embeddings corresponding to the input data.
    """
    if config["project"]["use_DB"]:
        # Load embeddings from Vector DB
        db = QdrantDB(
            collection_name=config["project"]["name"],
            host="http://ocean:6333"
            )
        X = db.database_access(
            df=df, plm_model=config["project"]["plm"]["plm_model"]
        )
    else:
        X = gen_embedding(
            sequences=df["sequence"].tolist(),
            plm_model=config["project"]["plm"]["plm_model"],
        )
    return X


def export_data(df, X_red, mst_array, linkage_array, output_dir="results") -> None:
    """
    Exports various data structures to files in specified formats for further use.
    """
    def sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares a Pandas DataFrame for saving as a Parquet file by sanitizing its columns.
        This function converts all columns with the data type 'object' to 'string' to avoid 
        potential type inference issues when using pyarrow for Parquet serialization.
        Args:
            df (pd.DataFrame): The input DataFrame to be sanitized.
        Returns:
            pd.DataFrame: A copy of the input DataFrame with 'object' columns converted to 'string'.
        """
        obj_cols = df.select_dtypes(include=["object"]).columns
        return df.copy().astype({col: "string" for col in obj_cols})

    # export data for minimal front end
    os.makedirs(output_dir, exist_ok=True)
    df_export = sanitize_for_parquet(df)
    df_export.to_parquet(os.path.join(output_dir, "df.parquet"), index=False)
    np.savez_compressed(os.path.join(output_dir, "X_red.npz"), X_red=X_red)
    np.savez_compressed(os.path.join(output_dir, "hdbscan_structures.npz"),
                        mst=mst_array,
                        linkage=linkage_array)
    
    # export data for user
    df.to_csv(os.path.join(output_dir + "data.csv"), index=False)
    df.to_csv(output_dir + "data.tsv", sep="\t", index=False)
    export_annotated_fasta(df=df, out_file=os.path.join(output_dir + "data.fasta"))


def main(app, config):
    df = load_and_preprocess(config)
    X = load_embeddings(config, df)

    # Clustering
    _mst, _linkage, df = perform_hdbscan_clustering(
        X,
        df,
        config["project"]["clustering"]["min_samples"],
        config["project"]["clustering"]["min_cluster_size"],
    )

    # Dimensionality reduction
    X_red = dimred_caller(
        X,
        config["project"]["dimred"]["method"],
        config["project"]["dimred"]["n_neighbors"],
        config["project"]["dimred"]["random_state"],
    )

    # save intermediates for external minimal dash version
    export_data(df, X_red, _mst, _linkage, output_dir=config["project"]["data"]["out_dir"])


    # Visualization
    fig = plot_2d(df, X_red, legend_attribute=config["project"]["plot_customizations"]["objective"])
    fig_mst = Figure(fig)  # copy required else fig will be modified by mst creation
    fig_cmst = Figure(fig)

    # Create page layouts
    dash.register_page(module="eda", name="Explanatory Data Analysis", layout=eda.layout(df))
    dash.register_page(
        module="dim",
        path="/",
        name="Protein Landscape",
        layout=dimred.layout(df, fig),
    )
    dash.register_page(
        module="mst", name="Connectivity", layout=mst.layout(_mst, df, X_red, fig_mst)
    )
    dash.register_page(module="slc", name="Phylogeny", layout=sl.layout(_linkage=_linkage, 
                                                                 df=df, 
                                                                 legend_attribute=config["project"]["plot_customizations"]["objective"]))
    
    # Register callbacks
    register_callbacks(app, df, X_red)

    # Centroid layouts: repeat clustering on only the centroids
    if set(df['cluster']) == {-1} and config["project"]["dimred"]["method"].upper() == "TSNE":  # skip centroid calculations if only outliers found or TSNE is used (no centroid projection possible)
        logging.error("No clusters found or t-SNE used, skipping centroid calculations.")
    else:
        # identify cluster centroids and their embeddings
        centroid_indices = df[df["marker_symbol"] == 'x'].index
        X_centroids = X[centroid_indices]
        X_red_centroids = X_red[centroid_indices]

        # Cluster centroids
        mst_centroids, linkage_centroids, df = perform_hdbscan_clustering(X_centroids, df, re_cluster=True)

        dash.register_page(
            module="cmst", 
            name="Centroid connectivity", 
            layout=mst.layout(mst_centroids, df, X_red_centroids, fig_cmst)
        )
        dash.register_page(module="cslc", 
                           name="Centroid Phylogeny", 
                           layout=sl_centroid.layout(_linkage=linkage_centroids, 
                                df=df[df['marker_symbol'] == 'x'], 
                                legend_attribute=config["project"]["plot_customizations"]["objective"]))

    # App layout with navigation links and page container
    app.layout = dbc.Container(
        [
            dbc.NavbarSimple(
                brand="Analysis results",
                color="primary",
                dark=True,
            ),
            html.Div(
                [
                    dcc.Store(
                        id="shared-data", data=[], storage_type="memory"
                    ),  # !saves table data from layouts via callbacks defined in the page layouts
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink(page["name"], href=page["path"]))
                            for page in dash.page_registry.values()
                        ],
                        pills=True,
                    ),
                    html.Hr(),
                    dash.page_container,  # Displays the content of the current page
                ]
            ),
        ],
        fluid=True,
    )


if __name__ == "__main__":
    import argparse

    from selectzyme.backend.utils import parse_args

    app = dash.Dash(
        __name__,
        use_pages=True,
        pages_folder="selectzyme/pages",
        assets_folder="selectzyme/assets",
        suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.BOOTSTRAP],  # Optional for styling
    )
    # server = app.server  # this line is only needed when deployed on a (public) server

    # CLI argument parsing
    config = parse_args()
    # Debugging way, only runs always the test_config.yml
    import yaml
    args = argparse.Namespace(config="results/input_configs/test_config.yml")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(app, config)
    app.run_server(host="127.0.0.1", port=config["project"]["port"], debug=False)
