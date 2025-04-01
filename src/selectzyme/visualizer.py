from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from selectzyme.customizations import set_columns_of_interest
from selectzyme.utils import run_time


@run_time
def plot_2d(
    df: pd.DataFrame,
    X_red: np.ndarray,
    legend_attribute: str,
) -> go.Figure:
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
            opacity = 0.5

        fig.add_trace(
            go.Scattergl(
                x=X_red[subset.index, 0],
                y=X_red[subset.index, 1],
                mode="markers",
                name=f"{str(attribute)[:40]} - {subset.shape[0]} entries",  # only show max. 40 characters in legend
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
                legend="legend",
            )
        )

    marker_legend_mapping = {
        "cross": "Brenda entries",
        "diamond": "UniProt entries",
        "circle": "Unannotated",
        "x": "Cluster centroid",
    }

    for symbol, label in marker_legend_mapping.items():
        fig.add_trace(
            go.Scattergl(
                x=[None],  # Empty traces for symbols. No toggle possible: todo: change this that toggle is possible
                y=[None],
                mode="markers",
                name=label,
                marker=dict(symbol=symbol, size=10, opacity=1),
                legend="legend2",  # Assign to second legend
            )
        )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            title=dict(text=legend_attribute),
            traceorder='normal'  # first occourence
        ),
        legend2=dict(
            title=dict(text="Marker Symbols"),
            orientation="h",
            entrywidth=70,
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig
