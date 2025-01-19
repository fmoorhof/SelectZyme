import logging

import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from cuml.cluster import (
    HDBSCAN,
    DBSCAN
)  # pip install hdbscan (the cuml is based on it else plotting can not be done direcly from the module)
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


def clustering_HDBSCAN(X, df: pd.DataFrame, min_samples: int = 30, min_cluster_size: int = 250, **kwargs):
    """
    Clustering of the embeddings with a Hierarchical Density Based clustering algorithm (HDBScan).
    # finished in 12 mins on 200k:)

    :param X: embeddings
    :param min_samples: amount of how many points shall be in a neighborhood of a point to form a cluster. 30 worked good for ec_only; 50 for 200k
    return: labels: cluster labels for each point
    """
    logging.info("Running HDBSCAN. This may take a while.")
    if X.shape[0] < min_samples:
        logging.error("The number of samples in X is less than min_samples. Please try a smaller value for min_samples.")
        raise ValueError("The number of samples in X is less than min_samples. Please try a smaller value for min_samples.")
    
    hdbscan = HDBSCAN(min_samples=min_samples, 
                      min_cluster_size=min_cluster_size, 
                      gen_min_span_tree=True, 
                      gen_condensed_tree=True, 
                      gen_single_linkage_tree_ = True,
                      **kwargs)  # todo: test: # condense_hierarchy: condenses the dendrogram to collapse subtrees containing less than min_cluster_size leaves, and returns an hdbscan.plots.CondensedTree object

    labels = hdbscan.fit_predict(X)

    G = hdbscan.minimum_spanning_tree_  # .to_networkx()  # # study:cuml/python/cuml/cuml/cluster/hdbscan/hdbscan.pyx: build_minimum_spanning_tree hdbscan.mst_dst, hdbscan.mst_weights
    Gsl = hdbscan.single_linkage_tree_  # .to_networkx()

    # plotting with default hdbscan reccomendation (matplotlib and hence interactivity missing) (remove when interactive plots enabled)
    # hdbscan.minimum_spanning_tree_.plot(edge_cmap='viridis',
    #                                   edge_alpha=0.6,
    #                                   node_size=80,
    #                                   edge_linewidth=2)
    # plt.savefig(f"datasets/mst.png", bbox_inches='tight')
    # plt.close()

    # deprecated when networkx replaced by SingleLinkageTree implementation (also remove df from function signature)
    # Annotate nodes with information from `df` (Assuming node indices in the graph match the DataFrame index)
    # Assuming nodes (NodeIDs) in G and Gls are the same -> performance enhancement (yes they match: nx.get_node_attributes(Gsl, "accession"))
    # for node in G.nodes():
    #     if node in df.index:
            # nx.set_node_attributes(Gsl, {node: df.loc[node].to_dict()})

    logging.info("HDBSCAN done")
    return labels, G, Gsl


def clustering_DBSCAN(X, eps: float = 1.0, min_samples: int = 1, **kwargs):
    """
    Clustering of the embeddings with a Density Based clustering algorithm (HDBScan).
    # finished in 12 mins on 200k:)

    :param X: embeddings
    :param min_samples: amount of how many points shall be in a neighborhood of a point to form a cluster. 30 worked good for ec_only; 50 for 200k
    :param kwargs: Additional parameters
    return: labels: cluster labels for each point
    """
    dbscan = DBSCAN(eps, min_samples=min_samples, **kwargs)
    labels = dbscan.fit_predict(X)
    logging.info("DBSCAN done")
    return labels


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