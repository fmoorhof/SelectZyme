from dash import html
from ydata_profiling import ProfileReport


def layout(df) -> html.Div:
    """Generates a Dash layout for the EDA using ydata"""
    df.pop('sequence')  # fix: column too long ValueError: Couldn't find space to draw. Either the Canvas size is too small or too much of the image is masked out.

    profile = ProfileReport(df, title="Profiling Report")
    profile.to_file("assets/census_report.html")

    return html.Div(
    children=[
        html.Iframe(
            src="assets/census_report.html",  # must be under assets/ to be properly served
            style={"height": "1080px", "width": "100%"},
        )
    ]
)
