import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram
from pandas import unique

from src.utils import run_time
from src.customizations import set_columns_of_interest


@run_time
def create_dendrogram(Z, df, legend_attribute: str = 'cluster'):
    P = dendrogram(Z, no_plot=True)
    icoord = np.array(P["icoord"])
    dcoord = np.array(P["dcoord"])
    leaves = P["leaves"]  # Indices of df rows corresponding to leaves

    layout = go.Layout(
        xaxis_title="Cluster/Variant",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis_title="Distance",
        showlegend=False,
    )
    fig = go.Figure(layout=layout)

    # Pre-calculate traces
    x_lines = _insert_separator(icoord)
    y_lines = _insert_separator(dcoord)
    # Add lines trace
    fig.add_trace(go.Scattergl(
        x=x_lines,
        y=y_lines,
        mode='lines',
        # line=dict(color=line_colors),  # not possible to pass list
        hoverinfo='none',
    ))

    # Pre-calculate markers
    marker_x = icoord[:, 0]  # always use left branch to place marker
    marker_y = dcoord[:, 1] - 0.001  # if set [0] or [1], hover breaks idk on this unexpected behaviour. 0.001 offset to avoid interference

    # Create a copy of the dataframe and sort it based on the indices 'leaves' deep copy needed, else not working!
    df_copy = df.iloc[leaves].copy(deep=True)
    df_copy = df_copy.sort_index()

    columns_of_interest = set_columns_of_interest(df_copy.columns)
    hover_text = ["<br>".join(f"{col}: {df_copy[col][i]}" for col in columns_of_interest) for i in range(len(df_copy))]

    # set color mapping for marker´s legend_attribute (mostly 'cluster')
    color_mapping = _value_to_color(df[legend_attribute])  # use 'old' unsorted df of other pages
    marker_colors = df_copy[legend_attribute].map(color_mapping).to_numpy()

    # Add markers trace using the sorted dataframe
    fig.add_trace(go.Scattergl(
        x=marker_x,
        y=marker_y,
        mode='markers',
        marker=dict(
            color=marker_colors,
            symbol=df_copy['marker_symbol'].to_numpy(),
            size=df_copy['marker_size'].to_numpy(),
            opacity=0.8
        ),
        customdata=df_copy['accession'].to_numpy(),
        text=np.array(hover_text),
        hoverinfo="text",
    ))

    return fig

def _value_to_color(values) -> dict:
    """
    Maps a list of values to a continuous color scale.

    Parameters:
    -----------
    values : array-like
        Values to map to a color scale.
    Returns:
    --------
        A dictionary mapping unique values to corresponding colors in the colormap.
    """
    # Get the unique values
    unique_values = unique(values)

    # Use Plotly's default qualitative color sequence
    default_colors = px.colors.qualitative.Plotly
    n_colors = len(default_colors)

    return {
        val: default_colors[i % n_colors]
        for i, val in enumerate(unique_values)
    }


def _insert_separator(arrays: np.ndarray) -> np.ndarray:
    # Append NaN to each array and concatenate
    return np.hstack([np.append(a, np.nan) for a in arrays])
    


if __name__ == "__main__":
    import hdbscan
    import numpy as np
    from sklearn.datasets import make_blobs
    import pandas as pd

    np.random.seed(42)
    sample_size = 11  # too big samples cause RecursionError but strangely not for my real datasets
    df = pd.DataFrame({
        'x': np.random.randn(sample_size),
        'y': np.random.randn(sample_size),
        'accession': np.arange(sample_size),
        'cluster': np.random.choice([-1, 1, 3], sample_size),
        'selected': np.random.choice([False, True], sample_size),
        'marker_symbol': np.random.choice(['circle', 'square', 'diamond', 'triangle-up'], sample_size),
        'marker_size': np.random.randint(10, sample_size, sample_size)
    })
    data, _ = make_blobs(n_samples=sample_size, n_features=2, centers=3, cluster_std=0.8, random_state=42)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)
    clusterer.fit(data)

    hover_text = ["<br>".join(f"{col}: {df[col][i]}" for col in df.columns) for i in range(len(df))]

    fig = create_dendrogram(Z=clusterer.single_linkage_tree_._linkage, df=df, hovertext=hover_text)
    fig.show()