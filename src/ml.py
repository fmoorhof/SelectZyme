import logging

import numpy as np
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
    pca = PCA(n_components=dimension, output_type="numpy")
    X_pca = pca.fit_transform(X)
    X_pca_centroid = pca.transform(X_centroids)
    variance = pca.explained_variance_ratio_ * 100
    variance = ["%.1f" % i for i in variance]  # 1 decimal only
    print(f"% Variance of the PCA components: {variance}")
    logging.info("PCA done")
    return X_pca, X_pca_centroid


def incremental_pca(X, X_centroids, dimension: int = 2, **kwargs):
    """Dimensionality reduction with Incremental PCA.
    Incremental PCA (Principal Component Analysis) is a variant of PCA that is designed to handle very large datasets that may not fit into memory.
    Standard PCA typically requires computing the covariance matrix of the entire dataset, which can be computationally expensive and memory-intensive, especially for large datasets. Incremental PCA, on the other hand, processes the dataset in smaller batches or chunks, allowing it to handle large datasets more efficiently.
    :param kwargs: Additional parameters"""
    ipca = IncrementalPCA(n_components=dimension, output_type="numpy", **kwargs)
    X_ipca = ipca.fit_transform(X)
    X_ipca_centroid = ipca.transform(X_centroids)
    variance = ipca.explained_variance_ratio_ * 100
    variance = ["%.1f" % i for i in variance]  # 1 decimal only
    print(f"% Variance of the PCA components: {variance}")
    logging.info("Incremental PCA done")
    return X_ipca, X_ipca_centroid


def truncated_svd(X, X_centroids, dimension: int = 2, **kwargs):
    """Dimensionality reduction with Truncated SVD.
    
    :param kwargs: Additional parameters
    """
    svd = TruncatedSVD(n_components=dimension, output_type="numpy", **kwargs)
    X_svd = svd.fit_transform(X)
    X_svd_centroid = svd.transform(X_centroids)
    logging.info("Truncated SVD done")
    return X_svd, X_svd_centroid


def tsne(X, dimension: int = 2, **kwargs):
    """Dimensionality reduction with tSNE.
    Currently TSNE supports n_components = 2; so only 2D plots are possible in May 2024!

    :param kwargs: Additional parameters for sklearn.manifold.TSNE
    """
    tsne = TSNE(n_components=dimension, **kwargs)
    X_tsne = tsne.fit_transform(X)
    logging.info("tSNE done. Cluster centroids can not meaningfully be transformed to 2D using tSNE. You might want to try openTSNE.")
    return X_tsne


def opentsne(X, X_centroids, dimension: int = 2, **kwargs):
    """Dimensionality reduction with open tSNE.
    Currently TSNE supports n_components = 2. Non GPU implementation but tsne.transform is possible to integrate cluster centroids.
    Despite, non GPU runtime is very good and scales less computationally complex in comparison to t-SNE.

    :param kwargs: Additional parameters for sklearn.manifold.TSNE
    """
    from openTSNE import TSNE
    tsne = TSNE(n_components=dimension, n_jobs=8, **kwargs)
    X_tsne = tsne.fit(X)
    X_tsne_centroid = X_tsne.transform(X_centroids)
    logging.info("Open tSNE done.")
    return X_tsne, X_tsne_centroid


# Dim reduced visualization with UMAP: super slow!! and .5GB output files wtf.
def umap(X, X_centroids, dimension: int = 2, **kwargs):
    """Dimensionality reduction with UMAP.
    :param kwargs: Additional parameters
    """
    umap = UMAP(n_components=dimension, **kwargs)  # unittest params: n_neighbors=10, min_dist=0.01
    X_umap = umap.fit_transform(X)
    X_umap_centroid = umap.transform(X_centroids)
    logging.info("UMAP done")
    return X_umap, X_umap_centroid