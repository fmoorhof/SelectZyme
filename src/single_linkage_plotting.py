"""Finally, simplest implementation, inspired by
plotly.figure_factory.create_dendrogram
but removed all additional functionalities and adapted template to my needs.
"""
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import plotly.graph_objects as go
import plotly.colors as pc


def create_dendrogram(Z, df, hovertext=None, legend_attribute: str = 'cluster'):
    P = dendrogram(Z, no_plot=True)
    icoord = np.array(P["icoord"])
    dcoord = np.array(P["dcoord"])

    layout = go.Layout(
        xaxis_title="Cluster/Variant",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis_title="Distance",
        showlegend=False,
        # legend_title_text=legend_attribute,
        )
    fig = go.Figure(layout=layout)

    # Get the color mapping and apply it to the DataFrame
    color_mapping = _value_to_color(df[legend_attribute])
    df['cluster_colors'] = df[legend_attribute].map(color_mapping)

    # create scatter traces
    for i in range(len(icoord)):  # perf: very very slow, breaking performance for large datasets
        fig.add_traces(go.Scattergl(
            x=icoord[i],
            y=dcoord[i],
            # name=str(legend_attribute),  # Legend name
            mode="lines+markers",
            line=dict(color="red" if df['selected'][i] == True else "black"),
            marker=dict(
                size=df['marker_size'][i],
                symbol=df['marker_symbol'][i],
                color=df['cluster_colors'][i],
                opacity=0.8
            ),
            customdata=df['accession'],
            text=hovertext[i],
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
    color_mapping : dict
        A dictionary mapping unique values to corresponding colors in the colormap.
    """
    # Normalize the values to [0, 1]
    unique_values = np.unique(values)
    norm = (unique_values - unique_values.min()) / (unique_values.max() - unique_values.min())
    
    # Get the Plotly colormap
    colorscale = pc.get_colorscale('Viridis')
    colormap_func = pc.sample_colorscale(colorscale, norm, low=0, high=1)
    
    # Map each unique value to a color
    color_mapping = dict(zip(unique_values, colormap_func))
    return color_mapping



if __name__ == "__main__":
    # from scipy.cluster import hierarchy as sch
    # # Example data
    # ytdist = np.array([662., 877., 255., 412., 996., 295., 468., 268.,
    #                    400., 754., 564., 138., 219., 869., 669.])
    # Z = sch.linkage(ytdist, "single")
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