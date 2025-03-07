from __future__ import annotations

import logging

import numpy as np
from cuml.cluster import HDBSCAN
from cuml.decomposition import PCA
from cuml.manifold import TSNE, UMAP

from src.utils import run_time


@run_time
def dimred_caller(
    X: np.ndarray, dim_method: str, n_neighbors: int = 15, random_state: int = 42, **kwargs
):
    dim_method = dim_method.upper()
    if dim_method == "PCA":
        X_red = pca(X, **kwargs)
    elif dim_method == "TSNE":
        X_red = tsne(X, perplexity=n_neighbors, random_state=random_state, **kwargs)
    elif dim_method == "OPENTSNE":
        X_red = opentsne(
            X, perplexity=n_neighbors, random_state=random_state, **kwargs
        )
    elif dim_method == "UMAP":
        X_red = umap(
            X, n_neighbors=n_neighbors, random_state=random_state, **kwargs
        )
    else:
        raise ValueError(
            f"Dimensionality reduction method {dim_method} not implemented. Choose from 'PCA', 'TSNE', 'openTSNE', 'UMAP'."
        )

    return X_red


def _indentify_centroid(model, X, cluster_id: int) -> int:
    """
    Identify the index of the real data point closest to the weighted centroid for a given cluster.
    
    :param model: The trained HDBSCAN model.
    :param X: The original dataset used for clustering.
    :param cluster_id: The cluster ID for which to find the closest point to the centroid.
    :return: Index of the closest data point in the original dataset.
    """
    mask = model.labels_ == cluster_id
    cluster_data = X[mask]
    cluster_indices = np.where(mask)[0]  # Get the original indices of the cluster members
    cluster_membership_strengths = model.probabilities_[mask]

    # Compute weighted centroid
    weighted_centroid = np.average(cluster_data, weights=cluster_membership_strengths, axis=0)

    distances = np.linalg.norm(cluster_data - weighted_centroid, axis=1)
    closest_point_index = np.argmin(distances)

    return cluster_indices[closest_point_index]


@run_time
def perform_hdbscan_clustering(X, df, min_samples: int = 30, min_cluster_size: int = 250, **kwargs):
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
        raise ValueError(
            "The number of samples in X is less than min_samples. Please try a smaller value for min_samples."
        )

    hdbscan = HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        gen_min_span_tree=True,
        **kwargs,
    )

    hdbscan.fit(X)
    labels = hdbscan.labels_

    G = hdbscan.minimum_spanning_tree_  # .to_networkx()  # # study:cuml/python/cuml/cuml/cluster/hdbscan/hdbscan.pyx: build_minimum_spanning_tree hdbscan.mst_dst, hdbscan.mst_weights
    Gsl = hdbscan.single_linkage_tree_  # .to_networkx()

    # Calculate centroids for each cluster
    centroid_indices = []
    for cluster_id in np.unique(labels):
        if cluster_id != -1:  # Skip noise cluster
            centroid_index = _indentify_centroid(hdbscan, X, cluster_id)
            centroid_indices.append(centroid_index)
    centroid_indices = np.array(centroid_indices)

    df["cluster"] = labels
    df.loc[centroid_indices, "marker_symbol"] = "x"

    return G, Gsl, df


def pca(X, dimension: int = 2, **kwargs):
    """Dimensionality reduction with PCA.
    :param kwargs: Additional parameters"""
    pca = PCA(n_components=dimension, output_type="numpy", **kwargs)
    X_pca = pca.fit_transform(X)
    variance = pca.explained_variance_ratio_ * 100
    variance = ["%.1f" % i for i in variance]  # 1 decimal only
    print(f"% Variance of the PCA components: {variance}")

    logging.info("PCA done")
    return X_pca


def tsne(X, dimension: int = 2, perplexity: int = 30, random_state: int = 42, **kwargs):
    """Dimensionality reduction with tSNE.
    Currently TSNE supports n_components = 2; so only 2D plots are possible in May 2024!
    Despite random seed, also tSNE looks not 100% reproducible. May be related to this bug report for UMAP:
    https://github.com/rapidsai/cuml/issues/5099

    :param kwargs: Additional parameters for sklearn.manifold.TSNE
    """
    tsne = TSNE(n_components=dimension, perplexity=perplexity, random_state=random_state, **kwargs)
    X_tsne = tsne.fit_transform(X)

    return X_tsne


def opentsne(X, dimension: int = 2, perplexity: int = 30, random_state: int = 42, **kwargs):
    """Dimensionality reduction with open tSNE.
    Currently TSNE supports n_components = 2. Non GPU implementation but tsne.transform is possible to integrate cluster centroids.
    Despite, non GPU runtime is very good and scales less computationally complex in comparison to t-SNE.

    :param kwargs: Additional parameters for sklearn.manifold.TSNE
    """
    from openTSNE import TSNE

    tsne = TSNE(n_components=dimension, perplexity=perplexity, n_jobs=8, random_state=random_state, **kwargs)
    X_tsne = tsne.fit(X)

    logging.info("Open tSNE done.")
    return X_tsne


def umap(
    X,
    dimension: int = 2,
    n_neighbors: int = 15,
    random_state: int = 42,
    **kwargs,
):
    """Dimensionality reduction with UMAP.
    Reproducibity warning: Despite random_state, UMAP multi-threaded has race conditions between the threads. Unfortunately this means that the randomness in UMAP outputs for the multi-threaded case depends not only on the random seed input, but also on race conditions between threads during optimization, over which no control can be had
    https://umap-learn.readthedocs.io/en/latest/reproducibility.html
    Also CuML´s UMAP has random_state problems:
    see unsolved bug issue: https://github.com/rapidsai/cuml/issues/5099
    suggested workaround is init='random'"""
    umap = UMAP(
        n_components=dimension,
        n_neighbors=n_neighbors,
        random_state=random_state,
        init="random",
        **kwargs,
    )  # , init="random" is walkaround until random_seed is fixed @ CuML; default metric='euclidean'
    X_umap = umap.fit_transform(X)

    logging.info("UMAP done")
    return X_umap
