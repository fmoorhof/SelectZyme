from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.customizations import set_columns_of_interest
from src.utils import run_time


@run_time
def plot_2d(
    df: pd.DataFrame,
    X_red: np.ndarray,
    X_red_centroids: np.ndarray,
    legend_attribute: str,
):
    """
    Plots a 2D scatter plot using Plotly based on the provided DataFrame and reduced dimensionality data.
    Parameters:
    df (pd.DataFrame): DataFrame containing the data to be plotted. It should include columns for the legend attribute, marker size, marker symbol, accession, and species.
    X_red (np.ndarray): 2D array with the reduced dimensionality data. The shape should be (n_samples, 2).
    legend_attribute (str): Column name in the DataFrame to be used for creating the legend.
    Returns:
    plotly.graph_objs._figure.Figure: A Plotly Figure object representing the 2D scatter plot.
    """
    fig = go.Figure()
    columns_of_interest = set_columns_of_interest(
        df.columns
    )  # Only show hover data for some df columns

    # Add a scatter trace for each unique attribute in the desired legend_attribute column
    for attribute in df[legend_attribute].unique():
        subset = df[df[legend_attribute] == attribute]

        # reduce opacity for large datasets
        opacity = 0.8
        if subset.size > 1000:
            opacity = 0.3

        fig.add_trace(
            go.Scattergl(
                x=X_red[subset.index, 0],
                y=X_red[subset.index, 1],
                mode="markers",
                name=str(attribute)[:40],  # only show max. 40 characters in legend
                marker=dict(
                    size=subset["marker_size"],
                    symbol=subset["marker_symbol"],
                    opacity=opacity,
                ),
                customdata=subset["accession"],
            hovertext=subset.apply(
                lambda row: "<br>".join(
                [f"{col}: {row[col]}" for col in columns_of_interest] +
                [f"x: {X_red[row.name, 0]}", f"y: {X_red[row.name, 1]}"]
                ),
                axis=1,
            ),
                hoverinfo="text",
            )
        )

    # add cluster centroids trace
    fig.add_trace(
        go.Scattergl(
            x=X_red_centroids[:, 0],
            y=X_red_centroids[:, 1],
            mode="markers",
            name="Cluster Centroids",  # Set the legend name
            marker=dict(size=10, symbol="x", color="red", opacity=0.3),
            hovertext=[
                f"Cluster {i} centroid" for i in range(X_red_centroids.shape[0])
            ],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        showlegend=True,
        legend_title_text=legend_attribute,
    )

    fig.write_html('results/dimred.html')
    return fig
