import logging

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

import pages.mst as mst
import pages.single_linkage as sl
import pages.dimred as dimred
import pages.eda as eda
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
    X = database_access(df, config['project']['name'], 
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
    dash.register_page('eda', name="Explanatory data analysis", layout=eda.layout(df))
    dash.register_page('dim', name="Dimensionality reduction and clustering", layout=dimred.layout(df, X_red, X_red_centroids))
    dash.register_page('sl', name="Phylogenetic Tree", layout=sl.layout(G=Gsl, df=df))    
    dash.register_page('mst', name="Minimal Spanning Tree", layout=mst.layout(G, df, X_red))

    # Register callbacks
    dimred.register_callbacks(app, df, X_red, X_red_centroids)

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

    # load real minimal data
    # args = argparse.Namespace(project_name='argparse_test_minimal', query_terms=["ec:1.13.11.85", "ec:1.13.11.84"], length='200 TO 601', custom_data_location="/raid/data/fmoorhof/PhD/SideShit/LCP/custom_seqs_no_signals.csv", dim_red='PCA', plm_model='prott5', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    # args = argparse.Namespace(project_name='argparse_test', query_terms=["ec:1.13.11.85", "latex clearing protein"], length='200 TO 601', custom_data_location="/raid/data/fmoorhof/PhD/SideShit/LCP/custom_seqs_no_signals.csv", dim_red='PCA', plm_model='esm1b', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    # args = argparse.Namespace(project_name='batch7', query_terms=["ec:1.14.11", "ec:1.14.20","xref%3Abrenda-1.14.11", "xref%3Abrenda-1.14.20", "IPR005123", "IPR003819", "IPR026992", "PF03171", "2OG-FeII_Oxy", "cd00250"], length='201 TO 500', custom_data_location="/raid/data/fmoorhof/PhD/Data/SKD022_2nd-order/custom_seqs_full.csv", dim_red='TSNE', plm_model='esm1b', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    # args = argparse.Namespace(project_name='lefos', query_terms=["ec:1.13.11.85", "ec:1.13.11.84"], length='200 TO 601', custom_data_location="/raid/data/fmoorhof/PhD/SideShit/LeFOS/custom_seqs.csv", dim_red='TSNE', plm_model='esm1b', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    # args = argparse.Namespace(project_name='lefos_prostt5', query_terms=["ec:1.13.11.85", "ec:1.13.11.84"], length='200 TO 601', custom_data_location="/raid/data/fmoorhof/PhD/SideShit/LeFOS/custom_seqs.csv", dim_red='TSNE', plm_model='prostt5', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    
    # args = argparse.Namespace(project_name='PapE', query_terms=["IPR001616", "IPR034720", "PF01771", "IPR011335"], length='350 TO 600', custom_data_location="/raid/data/fmoorhof/PhD/SideShit/PapE/custom_seqs.csv", dim_red='umap', plm_model='prott5', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    # args = argparse.Namespace(project_name='PapEC', query_terms=["IPR001616", "IPR034720", "PF01771"], length='350 TO 600', custom_data_location="/raid/data/fmoorhof/PhD/SideShit/PapE/custom_seqs.csv", dim_red='PCA', plm_model='prott5', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])

    # query_terms = ["ec:1.13.11.85", "xref%3Abrenda-1.13.11.85", "ec:1.13.11.87", "xref%3Abrenda-1.13.11.87", "ec:1.13.99.B1", "xref%3Abrenda-1.13.99.B1", "IPR037473", "IPR018713", "latex clearing protein"]  # define your query terms for UniProt here
    # args = argparse.Namespace(project_name='petase', query_terms=query_terms, length='50 TO 1020', custom_data_location='/raid/data/fmoorhof/PhD/Data/SKD021_Case_studies/PETase/pet_plasticDB_preprocessed.csv', dim_red='TSNE', plm_model='esm1b', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    # args = argparse.Namespace(project_name='petase2', query_terms=query_terms, length='50 TO 1020', custom_data_location='/raid/data/fmoorhof/PhD/Data/SKD021_Case_studies/PETase/pet_plasticDB_preprocessed_final.csv', dim_red='TSNE', plm_model='esm1b', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    # args = parse_args()

    # new argparsing implementation
    # CLI way
    config = parse_args()
    # Debugging way
    import yaml
    args = argparse.Namespace(config='results/upo.yml')
    with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    main(app=app)
    app.run_server(host="127.0.0.1", port=config['project']['port'], debug=False)
