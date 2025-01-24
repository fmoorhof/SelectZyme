import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import pages.mst as mst
import pages.single_linkage as sl
import pages.dimred as dimred
from src.main import parse_data, preprocessing, database_access, dimred_clust


# Initialize the Dash app
# app = dash.Dash(
#     __name__,
#     use_pages=True,  # Enables the multi-page functionality
#     suppress_callback_exceptions=True,
#     external_stylesheets=[dbc.themes.BOOTSTRAP],  # Optional for styling
# )
# server = app.server

def main(app):
    # backend calculations
    df = parse_data(args)
    df = preprocessing(df)
    X = database_access(df, args.project_name, args.plm_model)
    df, X_red, G, Gsl, X_red_centroids = dimred_clust(df, X, args.dim_red)

    # Create page layouts
    dash.register_page('dim', name="Dimensionality reduction and clustering", layout=dimred.layout(df, X_red, X_red_centroids))
    dash.register_page('mst', name="Minimal Spanning Tree", layout=mst.layout(G, df, X_red))
    dash.register_page('sl', name="Phylogenetic Tree", layout=sl.layout(G=Gsl, df=df, polar=False))  # todo: parse here truncation_mode and p
    # dimred_layout, dimred_register_callbacks = dimred.layout(df, X_red)
    # dimred_register_callbacks(app)  # Call the register_callbacks function to register the callbacks

    # Register callbacks
    dimred.register_callbacks(app, df, X_red, X_red_centroids)
    # other callbacks are not needed and can be removed theoretically in each layout: do this on cleanup
    # mst.register_callbacks(app, df)  # not needed any more since callback from dimred also apply on mst
    # sl.register_callbacks(app, df)  # not needed any more since callback from dimred also apply on mst


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
                    dcc.Store(id='shared-data', data=[]),  # Global shared data store
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
                    # dimred_layout  # insert dimensionality reduction layout from here
                ]
            ),
        ],
        fluid=True,
    )

    # Register callbacks for shared data
    register_shared_callbacks(app, df)


def register_shared_callbacks(app, df):
    # Callback to update shared data based on click interaction
    @app.callback(
        Output('shared-data', 'data'),
        Input('plot', 'clickData'),
        State('shared-data', 'data')
    )
    def update_shared_data(click_data, existing_data):
        # so far i dont know how this works here but if i comment all i get empy list always. with this entire code however i get duplicates that magically disappear after new selection
        if click_data is None:
            return existing_data

        # Extract accession from click data and lookup row in df
        accession = click_data['points'][0]['customdata']
        selected_row = df[df['accession'] == accession].iloc[0].to_dict()

        # Update shared data
        if existing_data is None:
            existing_data = []
        existing_data.append(selected_row)
        return existing_data

    # # Callback to display shared data in a table (optional for debugging or a shared page)
    # @app.callback(
    #     Output('shared-data-table', 'data'),
    #     Input('shared-data', 'data')
    # )
    # def update_shared_table(shared_data):
    #     return shared_data



if __name__ == "__main__":
    import argparse
    from src.main import parse_args

    app = dash.Dash(
        __name__,
        use_pages=True,  # Enables the multi-page functionality
        suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.BOOTSTRAP],  # Optional for styling
    )
    server = app.server

    # load real minimal data
    args = argparse.Namespace(project_name='argparse_test_minimal', query_terms=["ec:1.13.11.85", "ec:1.13.11.84"], length='200 TO 601', custom_data_location="/raid/data/fmoorhof/PhD/SideShit/LCP/custom_seqs_no_signals.csv", dim_red='TSNE', plm_model='esm1b', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    args = argparse.Namespace(project_name='argparse_test', query_terms=["ec:1.13.11.85", "latex clearing protein"], length='200 TO 601', custom_data_location="/raid/data/fmoorhof/PhD/SideShit/LCP/custom_seqs_no_signals.csv", dim_red='UMAP', plm_model='esm1b', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    # args = argparse.Namespace(project_name='batch7', query_terms=["ec:1.14.11", "ec:1.14.20","xref%3Abrenda-1.14.11", "xref%3Abrenda-1.14.20", "IPR005123", "IPR003819", "IPR026992", "PF03171", "2OG-FeII_Oxy", "cd00250"], length='201 TO 500', custom_data_location="/raid/data/fmoorhof/PhD/Data/SKD022_2nd-order/custom_seqs_full.csv", dim_red='TSNE', plm_model='esm1b', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    # args = argparse.Namespace(project_name='lefos', query_terms=["ec:1.13.11.85", "ec:1.13.11.84"], length='200 TO 601', custom_data_location="/raid/data/fmoorhof/PhD/SideShit/LeFOS/custom_seqs.csv", dim_red='TSNE', plm_model='esm1b', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    # args = argparse.Namespace(project_name='lefos_prostt5', query_terms=["ec:1.13.11.85", "ec:1.13.11.84"], length='200 TO 601', custom_data_location="/raid/data/fmoorhof/PhD/SideShit/LeFOS/custom_seqs.csv", dim_red='TSNE', plm_model='prostt5', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    # query_terms = ["ec:1.13.11.85", "xref%3Abrenda-1.13.11.85", "ec:1.13.11.87", "xref%3Abrenda-1.13.11.87", "ec:1.13.99.B1", "xref%3Abrenda-1.13.99.B1", "IPR037473", "IPR018713", "latex clearing protein"]  # define your query terms for UniProt here
    # args = argparse.Namespace(project_name='petase', query_terms=query_terms, length='50 TO 1020', custom_data_location='/raid/data/fmoorhof/PhD/Data/SKD021_Case_studies/PETase/pet_plasticDB_preprocessed.csv', dim_red='TSNE', plm_model='esm1b', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    # args = argparse.Namespace(project_name='petase2', query_terms=query_terms, length='50 TO 1020', custom_data_location='/raid/data/fmoorhof/PhD/Data/SKD021_Case_studies/PETase/pet_plasticDB_preprocessed_final.csv', dim_red='TSNE', plm_model='esm1b', out_dir='datasets/output/', df_coi=['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence'])
    # args = parse_args()

    main(app=app)
    app.run_server(host="0.0.0.0", port=8050, debug=False)
