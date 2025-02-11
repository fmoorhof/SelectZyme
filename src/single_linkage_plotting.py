import numpy as np
from scipy.cluster.hierarchy import dendrogram
import plotly.graph_objects as go
import plotly.colors as pc

from src.utils import run_time


@run_time
def create_dendrogram(Z, df, hovertext=None, legend_attribute: str = 'cluster'):
    P = dendrogram(Z, no_plot=True)
    icoord = np.array(P["icoord"])
    dcoord = np.array(P["dcoord"])

    layout = go.Layout(
        xaxis_title="Cluster/Variant",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis_title="Distance",
        showlegend=False,
    )
    fig = go.Figure(layout=layout)

    # Get the color mapping and apply it to the DataFrame
    color_mapping = _value_to_color(df[legend_attribute])
    df_colors = df[legend_attribute].map(color_mapping)

    # Prepare data for scatter plot
    x_lines = []
    y_lines = []
    # line_colors = []  # not implemented yet
    marker_x = []
    marker_y = []
    marker_colors = []
    marker_symbols = []
    marker_sizes = []
    customdata = []
    hovertexts = []

    for i in range(len(icoord)):
        x_lines.extend(icoord[i])
        x_lines.append(None)  # Add None to separate lines
        y_lines.extend(dcoord[i])
        y_lines.append(None)  # Add None to separate lines
        # line_colors.extend(["red" if df['selected'][i] else "black"] * 4)
        # line_colors.append(None)  # Add None to separate lines

        marker_x.append(icoord[i][0])  # always use left branch to place marker
        marker_y.append(dcoord[i][1] - 0.01)  # if set [0] or [1], hover breaks idk on this unexpected behaviour. 0.01 offset to avoid interference
        marker_colors.append(df_colors[i])
        marker_symbols.append(df['marker_symbol'][i])
        marker_sizes.append(df['marker_size'][i])
        customdata.append(df['accession'][i])
        hovertexts.append(hovertext[i])

    # Add lines trace
    fig.add_trace(go.Scattergl(
        x=x_lines,
        y=y_lines,
        mode='lines',
        # line=dict(color=line_colors),  # not possible to pass list
        hoverinfo='none',
    ))

    # Add markers trace
    fig.add_trace(go.Scattergl(
        x=marker_x,
        y=marker_y,
        mode='markers',
        marker=dict(
            color=marker_colors,
            symbol=marker_symbols,
            size=marker_sizes,
            opacity=0.8
        ),
        customdata=customdata,
        text=hovertexts,
        hoverinfo="text",
    ))

    return fig

def _value_to_color(values):
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
    # Normalize the values to [0, 1]
    unique_values = np.unique(values)
    if unique_values.max() == unique_values.min():
        norm = np.zeros_like(unique_values)
    else:
        norm = (unique_values - unique_values.min()) / (unique_values.max() - unique_values.min())
    
    # Get the Plotly colormap
    colorscale = pc.get_colorscale('Viridis')
    colormap_func = pc.sample_colorscale(colorscale, norm, low=0, high=1)

    return dict(zip(unique_values, colormap_func))  # Map each unique value to a color

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