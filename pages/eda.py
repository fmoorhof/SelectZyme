import logging

from dash import html
from ydata_profiling import ProfileReport

from src.customizations import set_columns_of_interest


def layout(df) -> html.Div:
    """Generates a Dash layout for the EDA using ydata"""
    logging.info("Generating EDA report. This may take a while...")
    columns_of_interest = set_columns_of_interest(df.columns)
    
    df_profile = df[columns_of_interest]  # discard columns that are not of interest such as marker_symbols etc.
    df_profile.drop(columns=['accession'], inplace=True)
    
    profile = ProfileReport(df_profile, title="Profiling Report", config_file="")  # empty string to fix docker TypeCheckError
    profile.to_file("assets/eda.html")

    return html.Div(
    children=[
        html.Iframe(
            src="assets/eda.html",  # must be under assets/ to be properly served
            style={"height": "1080px", "width": "100%"},
        )
    ]
)
