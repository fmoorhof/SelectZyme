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
from selectzyme.backend.customizations import custom_plotting
from selectzyme.backend.embed import gen_embedding
from selectzyme.backend.ml import dimred_caller, perform_hdbscan_clustering
from selectzyme.backend.parsing import parse_data
from selectzyme.backend.preprocessing import Preprocessing
from selectzyme.backend.utils import export_annotated_fasta
from selectzyme.frontend.visualizer import plot_2d
from selectzyme.pages.callbacks import register_callbacks


def parse_and_preprocess(config, existing_file) -> pd.DataFrame:
    """
    Parse and preprocess data based on the provided configuration (config.yml).
    This function parses data using the configuration parameters, applies 
    preprocessing if specified, and customizes the data for plotting.
    Returns:
        pandas.DataFrame: The processed and customized DataFrame.
    """
    df = parse_data(
        config["project"]["data"]["query_terms"],
        config["project"]["data"]["length"],
        config["project"]["data"]["custom_data_location"],
        existing_file,
        config["project"]["data"]["df_coi"],
    )

    if config["project"]["preprocessing"]:
        df = Preprocessing(df).preprocess()

    # Apply customizations
    df = custom_plotting(df=df, 
                         size=config["project"]["plot_customizations"]["size"], 
                         shape=config["project"]["plot_customizations"]["shape"])
    return df


def load_embeddings(df: pd.DataFrame, 
                    plm_model: str, 
                    embedding_file: str) -> np.ndarray:
    if os.path.exists(embedding_file):
        X = np.load(embedding_file)["X"]
        logging.info(f"Loaded embeddings from {embedding_file}")
    else:
        X = gen_embedding(
            sequences=df["sequence"].tolist(),
            plm_model=plm_model,
        )
        np.savez_compressed(embedding_file, X=X)
        logging.info(f"Saved embeddings to {embedding_file}")
    return X


def export_data(df: pd.DataFrame, 
                X_red: np.ndarray, 
                mst_array: np.ndarray, 
                linkage_array: np.ndarray,
                analysis_path: str) -> None:
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
    os.makedirs(analysis_path, exist_ok=True)
    df_export = sanitize_for_parquet(df)
    df_export.to_parquet(os.path.join(analysis_path, "df.parquet"), index=False)
    np.savez_compressed(os.path.join(analysis_path, "x_red_mst_slc.npz"),
                        X_red=X_red,
                        mst=mst_array,
                        linkage=linkage_array)
    
    # export data for user
    df.to_csv(os.path.join(analysis_path + "/data.csv"), index=False)
    df.to_csv(analysis_path + "/data.tsv", sep="\t", index=False)
    export_annotated_fasta(df=df, out_file=os.path.join(analysis_path + "/data.fasta"))


def main(app, config):
    # Backend
    analysis_path = os.path.join("results", config["project"]["name"])
    os.makedirs(analysis_path, exist_ok=True)

    df = parse_and_preprocess(config, existing_file=analysis_path + "/data.csv")
    X = load_embeddings(df, config["project"]["plm"]["plm_model"], embedding_file=os.path.join(analysis_path, "X.npz"))

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
    export_data(df, X_red, _mst, _linkage, analysis_path=analysis_path)

    # Repeat clustering on only the centroids
    if set(df['cluster']) != {-1}:
        # identify cluster centroids and their embeddings
        centroid_indices = df[df["marker_symbol"] == 'x'].index
        X_centroids = X[centroid_indices]
        X_red_centroids = X_red[centroid_indices]

        # Cluster centroids
        mst_centroids, linkage_centroids, df = perform_hdbscan_clustering(X_centroids, df, re_cluster=True)
    else:
        logging.info("Only outlier cluster found. Skipping centroid calculation.")


    # Frontend
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
    dash.register_page(module="slc", 
                       name="Phylogeny", 
                       layout=sl.layout(_linkage=_linkage, 
                                        df=df, 
                                        legend_attribute=config["project"]["plot_customizations"]["objective"]))

    # Register callbacks
    register_callbacks(app, df, X_red)

    # Create centroid layouts if cetroids are found
    if set(df['cluster']) != {-1}:
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

    # CLI argument parsing
    config = parse_args()
    # Debugging way, only runs always the test_config.yml
    import yaml
    args = argparse.Namespace(config="results/input_configs/test_config.yml")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(app, config)
    app.run_server(host="127.0.0.1", port=config["project"]["port"], debug=False)
