"""Finally, simplest implementation, inspired by
plotly.figure_factory.create_dendrogram
but removed all additional functionalities and adapted template to my needs.
"""
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import plotly.graph_objects as go

def create_dendrogram(Z, df, hovertext=None):
    P = dendrogram(Z, no_plot=True)
    icoord = np.array(P["icoord"])
    dcoord = np.array(P["dcoord"])

    # Create traces for the dendrogram
    traces = []
    for i in range(len(icoord)):
        text = hovertext[i] if hovertext else None

        trace = go.Scattergl(
            x=icoord[i],
            y=dcoord[i],
            mode="lines",
            line=dict(color="black"),
            customdata=df['accession'],
            text=text,
            hoverinfo="text",
        )
        traces.append(trace)
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        xaxis_title="Cluster/Variant",
        yaxis_title="Distance",
        showlegend=False,
    )

    return fig



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
        'accession': np.random.choice(['A', 'B', 'C'], sample_size),
        'cluster': np.random.choice(['A', 'B', 'C'], sample_size),
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