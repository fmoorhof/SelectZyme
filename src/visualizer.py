import logging

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from cuml.cluster import HDBSCAN
from cuml.decomposition import (
    PCA,
    TruncatedSVD,
    IncrementalPCA
)
from cuml.manifold import (
    TSNE,
    UMAP
)

from src.customizations import set_columns_of_interest


def _weighted_cluster_centroid(model, X, cluster_id) -> np.ndarray:
    """
    Calculate the weighted centroid for a given cluster.
    This function computes the weighted centroid of a cluster identified by `cluster_id` 
    using the provided clustering `model` and data `X`. The implementation is inspired 
    by HDBSCAN but adapted for use with CuML HDBSCAN, which does not expose the 
    `_raw_data` attribute.
    Parameters:
    model (HDBSCAN): The clustering model that has been fitted to the data.
    X (np.ndarray): The dataset used for clustering.
    cluster_id (int): The identifier of the cluster for which the centroid is to be calculated.
                      Note that `cluster_id` should not be -1, as this represents noise.
    Returns:
    np.ndarray: The weighted centroid of the specified cluster.
    Raises:
    ValueError: If `cluster_id` is -1, indicating a noise cluster.
    """
    if cluster_id == -1:
        raise ValueError("Cannot calculate centroid for noise cluster (-1).")
    
    mask = model.labels_ == cluster_id
    cluster_data = X[mask]  # model._raw_data[mask]  CuML HDBSCAN doesnt have _raw_data explosed but defined as GPUArray of X, called X_m and defined in from cuml.internals.input_utils import input_to_cuml_array
    cluster_membership_strengths = model.probabilities_[mask]
    
    return np.average(cluster_data, weights=cluster_membership_strengths, axis=0)


def clustering_HDBSCAN(X, min_samples: int = 30, min_cluster_size: int = 250, **kwargs):
    """
    Clustering of the embeddings with a Hierarchical Density Based clustering algorithm (HDBScan).
    # finished in 12 mins on 200k:)

    :param X: embeddings
    :param min_samples: amount of how many points shall be in a neighborhood of a point to form a cluster. 30 worked good for ec_only; 50 for 200k
    return: labels: cluster labels for each point
    """
    # todo: test: # condense_hierarchy: condenses the dendrogram to collapse subtrees containing less than min_cluster_size leaves, and returns an hdbscan.plots.CondensedTree object
    logging.info("Running HDBSCAN. This may take a while.")
    if X.shape[0] < min_samples:
        logging.error("The number of samples in X is less than min_samples. Please try a smaller value for min_samples.")
        raise ValueError("The number of samples in X is less than min_samples. Please try a smaller value for min_samples.")
    
    hdbscan = HDBSCAN(min_samples=min_samples, 
                      min_cluster_size=min_cluster_size, 
                      gen_min_span_tree=True, 
                      **kwargs)  

    hdbscan.fit(X)
    labels = hdbscan.labels_

    G = hdbscan.minimum_spanning_tree_  # .to_networkx()  # # study:cuml/python/cuml/cuml/cluster/hdbscan/hdbscan.pyx: build_minimum_spanning_tree hdbscan.mst_dst, hdbscan.mst_weights
    Gsl = hdbscan.single_linkage_tree_  # .to_networkx()

    # Calculate centroids for each cluster
    centroids = []
    for cluster_id in np.unique(labels):
        if cluster_id != -1:  # Skip noise cluster
            centroid = _weighted_cluster_centroid(hdbscan, X, cluster_id)
            centroids.append({
                'cluster': cluster_id,
                'x': centroid[0],
                'y': centroid[1] if X.shape[1] > 1 else None  # Handle 1D case
            })
    centroids_df = pd.DataFrame(centroids)

    logging.info("HDBSCAN done")
    return labels, G, Gsl, centroids_df


def pca(X, dimension: int = 2, **kwargs):
    """Dimensionality reduction with PCA.
    :param kwargs: Additional parameters"""
    pca = PCA(n_components=dimension, output_type="numpy")
    X_pca = pca.fit_transform(X)
    variance = pca.explained_variance_ratio_ * 100
    variance = ["%.1f" % i for i in variance]  # 1 decimal only
    print(f"% Variance of the PCA components: {variance}")
    logging.info("PCA done")
    return X_pca


def incremental_pca(X, dimension: int = 2, **kwargs):
    """Dimensionality reduction with Incremental PCA.
    Incremental PCA (Principal Component Analysis) is a variant of PCA that is designed to handle very large datasets that may not fit into memory.
    Standard PCA typically requires computing the covariance matrix of the entire dataset, which can be computationally expensive and memory-intensive, especially for large datasets. Incremental PCA, on the other hand, processes the dataset in smaller batches or chunks, allowing it to handle large datasets more efficiently.
    :param kwargs: Additional parameters"""
    ipca = IncrementalPCA(n_components=dimension, output_type="numpy", **kwargs)
    X_ipca = ipca.fit_transform(X)
    variance = ipca.explained_variance_ratio_ * 100
    variance = ["%.1f" % i for i in variance]  # 1 decimal only
    print(f"% Variance of the PCA components: {variance}")
    logging.info("Incremental PCA done")
    return X_ipca


def truncated_svd(X, dimension: int = 2, **kwargs):
    """Dimensionality reduction with Truncated SVD.
    
    :param kwargs: Additional parameters
    """
    svd = TruncatedSVD(n_components=dimension, output_type="numpy", **kwargs)
    X_svd = svd.fit_transform(X)
    logging.info("Truncated SVD done")
    return X_svd


def tsne(X, dimension: int = 2, **kwargs):
    """Dimensionality reduction with tSNE.
    Currently TSNE supports n_components = 2; so only 2D plots are possible in May 2024!

    :param kwargs: Additional parameters for sklearn.manifold.TSNE
    """
    tsne = TSNE(n_components=dimension, **kwargs)
    X_tsne = tsne.fit_transform(X)
    logging.info("tSNE done")
    return X_tsne


# Dim reduced visualization with UMAP: super slow!! and .5GB output files wtf.
def umap(X, dimension: int = 2, **kwargs):
    """Dimensionality reduction with UMAP.
    :param kwargs: Additional parameters
    """
    umap = UMAP(n_components=dimension, **kwargs)  # unittest params: n_neighbors=10, min_dist=0.01
    X_umap = umap.fit_transform(X)
    logging.info("UMAP done")
    return X_umap


def plot_2d(df, X_red, legend_attribute: str):
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

    # Add a scatter trace for each unique value in the legend_attribute column
    for value in df[legend_attribute].unique():
        subset = df[df[legend_attribute] == value]

        columns_of_interest = set_columns_of_interest(df.columns)  # Only show hover data for some df columns

        fig.add_trace(go.Scattergl(
            x=X_red[subset.index, 0],
            y=X_red[subset.index, 1],
            mode='markers',
            name=str(value),  # Legend name
            marker=dict(
                size=subset['marker_size'],
                symbol=subset['marker_symbol'],
                opacity=0.5
            ),
            customdata=subset['accession'],
            hovertext=subset.apply(lambda row: '<br>'.join([f'{col}: {row[col]}' for col in columns_of_interest]), axis=1),
            hoverinfo='text'
        ))

    fig.update_layout(
        showlegend=True,
        legend_title_text=legend_attribute
    )
    # fig.write_html(f'datasets/test_landscape.html')
    return fig



if __name__ == "__main__":
    raise NotImplementedError("This script is not ready yet to run directly from here.")