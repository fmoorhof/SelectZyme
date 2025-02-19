import logging

import numpy as np
from cuml.cluster import HDBSCAN
from cuml.decomposition import PCA
from cuml.manifold import (
    TSNE,
    UMAP
)

from src.utils import run_time


@run_time
def dimred_caller(X, X_centroids, dim_method, n_neighbors: int = 15, random_state: int = 42, **kwargs):
    dim_method = dim_method.upper()
    if dim_method == 'PCA':
        X_red, X_red_centroids = pca(X, X_centroids, **kwargs)
    elif dim_method == 'TSNE':
        X_red, X_red_centroids = tsne(X, random_state=random_state, **kwargs)
    elif dim_method == 'OPENTSNE':
        X_red, X_red_centroids = opentsne(X, X_centroids, random_state=random_state, **kwargs)
    elif dim_method == 'UMAP':
        X_red, X_red_centroids = umap(X, X_centroids, n_neighbors=n_neighbors, random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Dimensionality reduction method {dim_method} not implemented. Choose from 'PCA', 'TSNE', 'openTSNE', 'UMAP'.")
    
    return X_red, X_red_centroids


def _weighted_cluster_centroid(model, X, cluster_id: int) -> np.ndarray:
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
    np.ndarray: The weighted centroid of the specified cluster..
    """
    mask = model.labels_ == cluster_id
    cluster_data = X[mask]  # model._raw_data[mask]  CuML HDBSCAN doesnt have _raw_data explosed but defined as GPUArray of X, called X_m and defined in from cuml.internals.input_utils import input_to_cuml_array
    cluster_membership_strengths = model.probabilities_[mask]
    
    return np.average(cluster_data, weights=cluster_membership_strengths, axis=0)


@run_time
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
            centroids.append(centroid)
    X_centroids = np.array(centroids)

    logging.info("HDBSCAN done")
    return labels, G, Gsl, X_centroids


def pca(X, X_centroids, dimension: int = 2, **kwargs):
    """Dimensionality reduction with PCA.
    :param kwargs: Additional parameters"""
    pca = PCA(n_components=dimension, output_type="numpy", **kwargs)
    X_pca = pca.fit_transform(X)
    variance = pca.explained_variance_ratio_ * 100
    variance = ["%.1f" % i for i in variance]  # 1 decimal only
    print(f"% Variance of the PCA components: {variance}")

    if X_centroids.size != 0:
        X_pca_centroid = pca.transform(X_centroids)
    else:
        X_pca_centroid = np.empty((0, 2))
        logging.info("HDBSCAN cluster parameter yielded no clusters. Concludingly, no cluster centroids are returned.")

    logging.info("PCA done")
    return X_pca, X_pca_centroid



def tsne(X, dimension: int = 2, random_state: int = 42, **kwargs):
    """Dimensionality reduction with tSNE.
    Currently TSNE supports n_components = 2; so only 2D plots are possible in May 2024!
    Despite random seed, also tSNE looks not 100% reproducible. May be related to this bug report for UMAP:
    https://github.com/rapidsai/cuml/issues/5099

    :param kwargs: Additional parameters for sklearn.manifold.TSNE
    """
    tsne = TSNE(n_components=dimension, random_state=random_state, **kwargs)
    X_tsne = tsne.fit_transform(X)
    X_tsne_centroid = np.empty((0, 2))
    logging.info("tSNE done. Cluster centroids can not meaningfully be transformed to 2D using tSNE. You might want to try openTSNE.")
    return X_tsne, X_tsne_centroid


def opentsne(X, X_centroids, dimension: int = 2, random_state: int = 42, **kwargs):
    """Dimensionality reduction with open tSNE.
    Currently TSNE supports n_components = 2. Non GPU implementation but tsne.transform is possible to integrate cluster centroids.
    Despite, non GPU runtime is very good and scales less computationally complex in comparison to t-SNE.

    :param kwargs: Additional parameters for sklearn.manifold.TSNE
    """
    from openTSNE import TSNE
    tsne = TSNE(n_components=dimension, n_jobs=8, random_state=random_state, **kwargs)
    X_tsne = tsne.fit(X)

    if X_centroids.size != 0:
        X_tsne_centroid = X_tsne.transform(X_centroids)
    else:
        X_tsne_centroid = np.empty((0, 2))
        logging.info("HDBSCAN cluster parameter yielded no clusters. Concludingly, no cluster centroids are returned.")
    
    logging.info("Open tSNE done.")
    return X_tsne, X_tsne_centroid


def umap(X, X_centroids, dimension: int = 2, n_neighbors: int = 15, random_state: int = 42, **kwargs):
    """Dimensionality reduction with UMAP.
    Reproducibity warning: Despite random_state, UMAP multi-threaded has race conditions between the threads. Unfortunately this means that the randomness in UMAP outputs for the multi-threaded case depends not only on the random seed input, but also on race conditions between threads during optimization, over which no control can be had
    https://umap-learn.readthedocs.io/en/latest/reproducibility.html
    Also CuML´s UMAP has random_state problems:
    see unsolved bug issue: https://github.com/rapidsai/cuml/issues/5099
    suggested workaround is init='random'"""
    umap = UMAP(n_components=dimension, n_neighbors=n_neighbors, random_state=random_state, init="random", **kwargs)  # , init="random" is walkaround until random_seed is fixed @ CuML; default metric='euclidean'
    X_umap = umap.fit_transform(X)
    
    if X_centroids.size != 0:
        X_umap_centroid = umap.transform(X_centroids)
    else:
        X_umap_centroid = np.empty((0, 2))
        logging.info("HDBSCAN cluster parameter yielded no clusters. Concludingly, no cluster centroids are returned.")

    logging.info("UMAP done")
    return X_umap, X_umap_centroid
