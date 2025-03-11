from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO)

import os

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from plotly.graph_objects import Figure

import pages.dimred as dimred
import pages.eda as eda
import pages.mst as mst
import pages.single_linkage as sl
import pages.slc_centroid as sl_centroid
from pages.callbacks import register_callbacks
from selectzyme.customizations import custom_plotting
from selectzyme.embed import gen_embedding
from selectzyme.ml import dimred_caller, perform_hdbscan_clustering
from selectzyme.preprocessing import Preprocessing
from selectzyme.utils import export_annotated_fasta, parse_data
from selectzyme.vector_db import QdrantDB
from selectzyme.visualizer import plot_2d


def load_and_preprocess(config):
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


def main(app, config):
    export_path = os.path.join(config["project"]["data"]["out_dir"] + config["project"]["name"])

    df = load_and_preprocess(config)
    X = load_embeddings(config, df)

    # Clustering
    G, Gsl, df = perform_hdbscan_clustering(
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

    # Perf: create DimRed and MST plot only once
    fig = plot_2d(df, X_red, legend_attribute="cluster")
    fig_mst = Figure(fig)  # copy required else fig will be modified by mst creation
    fig_cmst = Figure(fig)

    # Create page layouts
    dash.register_page("eda", name="Explanatory Data Analysis", layout=eda.layout(df, out_file=export_path + "_eda.html"))
    dash.register_page(
        "dim",
        name="Dimensionality Reduction and Clustering",
        layout=dimred.layout(df, fig),
    )
    dash.register_page(
        "mst", name="Minimal Spanning Tree (MST)", layout=mst.layout(G, df, X_red, fig_mst)
    )
    dash.register_page("slc", name="Phylogram", layout=sl.layout(G=Gsl, df=df, out_file=export_path + "_slc.html"))

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
        G_centroids, Gsl_centroids, df = perform_hdbscan_clustering(X_centroids, df, re_cluster=True)

        dash.register_page(
            "cmst", name="Centroid MST", layout=mst.layout(G_centroids, df, X_red_centroids, fig_cmst)
        )
        dash.register_page("cslc", name="Centroid Phylogram", layout=sl_centroid.layout(G=Gsl_centroids, df=df[df['marker_symbol'] == 'x'], out_file=export_path + "_cslc.html"))
        fig_cmst.write_html(export_path + "_cmst.html")

    # export data and plots
    df.to_csv(export_path + ".csv", index=False)
    export_annotated_fasta(df=df, out_file=export_path + ".fasta")
    fig.write_html(export_path + "_dimred.html")
    fig_mst.write_html(export_path + "_mst.html")


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

    from selectzyme.utils import parse_args

    app = dash.Dash(
        __name__,
        use_pages=True,
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
