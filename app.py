from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO)

import os

import argparse
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

import selectzyme.pages.dimred as dimred
import selectzyme.pages.eda as eda
from selectzyme.backend.embed import load_embeddings
from selectzyme.backend.ml import dimred_caller, perform_hdbscan_clustering
from selectzyme.backend.predict_ec import run_clean_inference_with_embeddings
from selectzyme.backend.predict_solubility import get_preds
from selectzyme.backend.utils import export_data, parse_and_preprocess
from selectzyme.frontend.mst_plotting import MinimumSpanningTree
from selectzyme.frontend.single_linkage_plotting import create_dendrogram
from selectzyme.frontend.visualizer import plot_2d
from selectzyme.pages.callbacks import register_callbacks


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

    # predict EC numbers with CLEAN
    if config["project"]["plm"]["plm_model"] == "esm1b" and config["project"]["predictions"]["ec"] == True:
        data_path = "../CLEAN/app/data/"
        desired_split = "100"
        df_clean = run_clean_inference_with_embeddings(
            sequence_label_esm_emb_dict={row["accession"]: X[i] for i, row in df.iterrows()},
            emb_train_path=f'{data_path}pretrained/{desired_split}.pt',
            ec_csv_path=f'{data_path}split{desired_split}.csv',
            model_ckpt_path=f'{data_path}pretrained/split{desired_split}.pth',
            out_csv=analysis_path + "/clean_predictions.csv",
            gmm=f'{data_path}pretrained/gmm_ensumble.pkl',
            device="cpu",
        )
        
        df = df.merge(df_clean[["accession", "CLEAN_EC_pred", "CLEAN_probability"]], on="accession", how="left")

    # predict solubility and usability with NetSolP
    if config["project"]["predictions"]["solubility"] == True:
        logging.info("Running NetSolP predictions. This might take a long time.")
        # hard coded configurations to pass to NetSolP package
        parser = argparse.ArgumentParser()
        parser.add_argument("--MODEL_TYPE", default="ESM1b")
        parser.add_argument("--MODELS_PATH", 
                            default="NetSolP-1.0/PredictionServer/models")  # /scratch/global_1/fmoorhof/NetSolP/models/
        parser.add_argument("--NUM_THREADS", default=os.cpu_count(), type=int)
        parser.add_argument(
            "--PREDICTION_TYPE",
            default="SU",
            choices=['S', 'U', 'SU'],
            type=str,
            help="Either Solubility(S), Usability(U) or Both"
        )
        args = parser.parse_args()

        df = get_preds(df=df, args=args)

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


    # Frontend / Middleware
    # Create all plots
    fig = plot_2d(df, X_red, legend_attribute=config["project"]["plot_customizations"]["objective"])
    mst_obj = MinimumSpanningTree(_mst, df, X_red, fig)
    fig_mst = mst_obj.plot_mst_in_dimred_landscape()
    # fig = mst.plot_mst_force_directed(G)  # deprecated. usage of graph G not supported any more, remove functionality in near future
    fig_slc = create_dendrogram(Z=_linkage, 
                                df=df, 
                                legend_attribute=config["project"]["plot_customizations"]["objective"])

    if set(df['cluster']) != {-1}:
        # Centroid Minimal Spanning Tree
        mst_obj = MinimumSpanningTree(mst_centroids, df, X_red_centroids, fig)
        fig_cmst = mst_obj.plot_mst_in_dimred_landscape()

        # Centroid Phylogeny
        fig_cslc = create_dendrogram(Z=linkage_centroids, 
                                     df=df[df['marker_symbol'] == 'x'], 
                                     legend_attribute=config["project"]["plot_customizations"]["objective"])
        

    # Register pages
    dash.register_page(module="eda", name="Exploratory Data Analysis", layout=eda.layout(df))
    dash.register_page(
        module="dim",
        path="/",
        name="Protein Landscape",
        layout=dimred.layout(df.columns, fig, dropdown=True),
    )
    dash.register_page(module="mst", name="Connectivity", layout=dimred.layout(df.columns, fig_mst))
    dash.register_page(module="slc", name="Phylogeny", layout=dimred.layout(df.columns, fig_slc))

    # Create centroid layouts if cetroids are found
    if set(df['cluster']) != {-1}:
        # Centroid Minimal Spanning Tree
        dash.register_page(
            module="cmst", 
            name="Centroid connectivity", 
            layout=dimred.layout(df.columns, fig_cmst)
        )

        # Centroid Phylogeny
        dash.register_page(module="cslc", 
                           name="Centroid Phylogeny", 
                           layout=dimred.layout(df.columns, fig_cslc))
        
    # Register callbacks
    register_callbacks(app, df, X_red)

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
