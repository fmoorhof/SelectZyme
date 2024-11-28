from src.dash_app import run_dash_app


def layout(df, X_red, dim_red, project_name):
    layout, register_callbacks = run_dash_app(df, X_red, dim_red, project_name)
    return layout, register_callbacks
