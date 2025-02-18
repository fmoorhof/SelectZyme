import logging
logging.basicConfig(level=logging.INFO)

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

import pages.mst as mst
import pages.single_linkage as sl
import pages.dimred as dimred
import pages.eda as eda
from pages.callbacks import register_callbacks
from src.utils import parse_data, database_access
from src.preprocessing import Preprocessing
from src.ml import dimred_caller, clustering_HDBSCAN
from src.customizations import custom_plotting


def main(app):
    # backend calculations
    df = parse_data(config['project']['name'], 
                    config['project']['data']['query_terms'], 
                    config['project']['data']['length'], 
                    config['project']['data']['custom_data_location'], 
                    config['project']['data']['out_dir'], 
                    config['project']['data']['df_coi'])
    logging.info(f"df columns have the dtypes: {df.dtypes}")

    df = Preprocessing(df).preprocess()

    # Load embeddings from Vector DB
    X = database_access(df, 
                        config['project']['name'], 
                        config['project']['plm']['plm_model'])

    # Clustering
    labels, G, Gsl, X_centroids = clustering_HDBSCAN(X, 
                                                     config['project']['clustering']['min_samples'], 
                                                     config['project']['clustering']['min_cluster_size'])
    df['cluster'] = labels
    df = custom_plotting(df)

    # Dimensionality reduction
    X_red, X_red_centroids = dimred_caller(X, 
                                           X_centroids, 
                                           config['project']['dimred']['method'],
                                           config['project']['dimred']['n_neighbors'],
                                           config['project']['dimred']['random_state'])

    # Create page layouts
    dash.register_page('eda', name="Explanatory Data Analysis", layout=eda.layout(df))
    dash.register_page('dim', name="Dimensionality Reduction and Clustering", layout=dimred.layout(df, X_red, X_red_centroids))
    dash.register_page('slc', name="Phylogenetic Tree", layout=sl.layout(G=Gsl, df=df))    
    dash.register_page('mst', name="Minimal Spanning Tree", layout=mst.layout(G, df, X_red))

    # Register callbacks
    register_callbacks(app, df, X_red, X_red_centroids)

    # Layout with navigation links and page container
    app.layout = dbc.Container(
        [
            dbc.NavbarSimple(
                brand="Analysis results",
                color="primary",
                dark=True,
            ),
            html.Div(
                [
                    dcc.Store(id='shared-data', data=[], storage_type='memory'),  # !saves table data from layouts via callbacks defined in the page layouts
                    dbc.Nav(
                        [
                            dbc.NavItem(
                                dbc.NavLink(page["name"], href=page["path"])
                            )
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
    from src.utils import parse_args

    app = dash.Dash(
        __name__,
        use_pages=True,  # Enables the multi-page functionality
        suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.BOOTSTRAP],  # Optional for styling
    )
    # server = app.server  # this line is only needed when deployed on a (public) server

    # CLI argument parsing
    config = parse_args()
    # Debugging way, only runs always the test_config.yml
    import yaml
    args = argparse.Namespace(config='results/test_config.yml')
    with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    main(app=app)
    app.run_server(host="127.0.0.1", port=config['project']['port'], debug=False)
