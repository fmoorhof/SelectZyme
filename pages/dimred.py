from src.dash_app import run_dash_app


def layout(df, X_red):
    layout, register_callbacks = run_dash_app(df, X_red)
    return layout, register_callbacks
