"""
This file is accessing the Qdrant vector database for further analysis. On hughe datasets the CPU implemented tools show a very long runtime. Therefore, this script is
aiming to perform downstream analysis with GPU only.
 
downstream analysis:
clustering
dimensionality reduction
visualization
 
Runtime < 30 mins for 200k sequences
 
Execution hint: RAPIDSAI was not installable in DeepChem, so i execute this script in a seperate docker container that only contains the RAPIDSAI tools.
The container name is fmoorhof_rapidsai and the ID: 2cee57c21810
rapidsai/rapidsai:cuda11.5-base-centos7-py3.9
"""
import logging

import pandas as pd
import plotly.express as px

import cudf
from cuml.cluster import (
    HDBSCAN,
    DBSCAN
)  # pip install hdbscan (the cuml is based on it else plotting can not be done direcly from the module)
from cuml.decomposition import PCA
from cuml.manifold import (
    TSNE,
    UMAP
)


def clustering_HDBSCAN(X, min_samples: int = 30, min_cluster_size: int = 250):
    """
    Clustering of the embeddings with a Hierarchical Density Based clustering algorithm (HDBScan).
    # finished in 12 mins on 200k:)

    :param X: embeddings
    :param min_samples: amount of how many points shall be in a neighborhood of a point to form a cluster. 30 worked good for ec_only; 50 for 200k
    return: labels: cluster labels for each point
    """
    if X.shape[0] < min_samples:
        logging.error("The number of samples in X is less than min_samples. Please try a smaller value for min_samples.")
        raise ValueError("The number of samples in X is less than min_samples. Please try a smaller value for min_samples.")
    
    hdbscan = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    labels = hdbscan.fit_predict(X)
    logging.info(f"HDBSCAN done")
    return labels


def clustering_DBSCAN(X, eps: float = 1.0, min_samples: int = 1):
    """
    Clustering of the embeddings with a Density Based clustering algorithm (HDBScan).
    # finished in 12 mins on 200k:)

    :param X: embeddings
    :param min_samples: amount of how many points shall be in a neighborhood of a point to form a cluster. 30 worked good for ec_only; 50 for 200k
    return: labels: cluster labels for each point
    """
    dbscan = DBSCAN(eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    logging.info(f"DBSCAN done")
    return labels


def pca(X, dimension: int = 2):
    """Dimensionality reduction with PCA."""
    pca = PCA(n_components=dimension, output_type="numpy")
    X_pca = pca.fit_transform(X)
    variance = pca.explained_variance_ratio_ * 100
    variance = ["%.1f" % i for i in variance]  # 1 decimal only
    print(f"% Variance of the PCA components: {variance}")
    logging.info(f"PCA done")
    return X_pca


def tsne(X, dimension: int = 2):
    """Dimensionality reduction with tSNE."""
    tsne = TSNE(n_components=dimension, random_state=42)
    X_tsne = tsne.fit_transform(X)
    logging.info(f"tSNE done")
    return X_tsne


# Dim reduced visualization with UMAP: super slow!! and .5GB output files wtf.
def umap(X, dimension: int = 2, n_neighbors: int = 15):
    """Dimensionality reduction with UMAP."""
    umap = UMAP(n_neighbors=n_neighbors, n_components=dimension, random_state=42)  # unittest params: n_neighbors=10, min_dist=0.01
    X_umap = umap.fit_transform(X)
    logging.info(f"UMAP done")
    return X_umap


def custom_plotting(df, labels):
    """Modify the df before plotting."""
    df['cluster'] = labels  # add cluster labels to df (from clustering)

    # Create new columns 'marker_size' and 'marker_symbol' based on a condition
    df['EC number'] = df['EC number'].fillna('0.0.0.0')  # replace empty ECs because they will not get plottet (if color='EC number')
    # df['BRENDA'] = df['BRENDA'].fillna('')  # replace empty ECs because they will not get plottet (if color='EC number')
    values_to_replace = ['NA', '', '0']
    df['BRENDA'] = df['BRENDA'].replace(values_to_replace, '')
    
    values_to_replace = ['1.14.11.-', '1.14.20.-']
    df['EC number'] = df['EC number'].replace(values_to_replace, '0.0.0.1')
    values_to_replace = ['1.-.-.-']
    df['EC number'] = df['EC number'].replace(values_to_replace, '0.0.0.0')
    
    if isinstance(df, cudf.DataFrame):  # fix for AttributeError: 'Series' object has no attribute 'to_pandas' (cudf vs. pandas)
        condition = (df['BRENDA'].to_pandas() != '') 
        condition2 = (df['EC number'].to_pandas() != '0.0.0.0')
    else:  # pandas DataFrame
        condition = (df['BRENDA'] != '') 
        condition2 = (df['EC number'] != '0.0.0.0')

    df['marker_size'] = 5
    df['marker_symbol'] = 'circle'
    df.loc[condition2, 'marker_size'] = 6  # Set to other value for data points that meet the condition
    df.loc[condition2, 'marker_symbol'] = 'diamond'
    df.loc[condition, 'marker_size'] = 18
    df.loc[condition, 'marker_symbol'] = 'cross'
    # df.loc[condition & condition2, 'marker_size'] = 14  # 2 conditions possible
    
    # build Brenda URLs
    df['BRENDA URL'] = [
        f"https://www.brenda-enzymes.org/enzyme.php?ecno={ec.split(';')[0]}&UniProtAcc={entry}&OrganismID={organism}"
        if pd.notna(ec)
        else pd.NA  # Fill with NaN for rows where BRENDA is NaN
        for ec, entry, organism in zip(df['BRENDA'].values, df['Entry'].values, df['Organism (ID)'].values)  # values_host with cudf
    ]
    
    # alphabetically sort df based on EC numbers (for nicer legend)
    df = df.sort_values(by=['EC number'])

    return df


def plot_2d(df, X_red, collection_name: str, method: str):
    """Plot the results, independent of the dimensionality method used. Output files are written to the Output folder.

    :param df: dataframe containing the annoattions
    :param X_red: dimensionality reduced embeddings
    :param collection_name: name of the collection/dataset
    :param method: dimensionality reduction method used"""
    cols = df.columns.values.tolist()
    # cols = cols[0:-2]  # do not provide sequence
    fig = px.scatter(df, x=X_red[:, 0], y=X_red[:, 1],  # X_umap[0].to_numpy()?
                     color='EC number', # color='cluster'
                 title=f'2D {method} on dataset {collection_name}',
                 hover_data=cols,
                 opacity=0.5,
                 color_continuous_scale=px.colors.sequential.Viridis,  # _r = reversed  # color_discrete_sequence=px.colors.sequential.Viridis,
                 )
 
    fig.update_traces(marker=dict(size=df['marker_size'].to_numpy(), symbol=df['marker_symbol'].to_numpy()))
    fig.write_html(f'datasets/output/{collection_name}_2d_{method}.html')
    logging.info(f'{method} 2D plot completed.')


def plot_3d(df, X_red, collection_name: str, method: str):
    """Plot the results, independent of the dimensionality method used. Output files are written to the Output folder.

    :param df: dataframe containing the annoattions
    :param X_red: dimensionality reduced embeddings
    :param collection_name: name of the collection/dataset
    :param method: dimensionality reduction method used"""
    cols = df.columns.values.tolist()
    # cols = cols[0:-2]  # do not provide sequence
    fig = px.scatter_3d(df, x=X_red[:, 0], y=X_red[:, 1], z=X_red[:, 2],  # X_umap[0].to_numpy()?
                        color='EC number', # color='cluster'
                 title=f'3D {method} on dataset {collection_name}',
                 hover_data=cols,
                 opacity=0.5,
                 color_continuous_scale=px.colors.sequential.Viridis,  # _r = reversed  # color_discrete_sequence=px.colors.sequential.Viridis,
                 )
 
    fig.update_traces(marker=dict(size=df['marker_size'].to_numpy(), symbol=df['marker_symbol'].to_numpy()))
    fig.write_html(f'datasets/output/{collection_name}_3d_{method}.html')
    logging.info(f'{method} 3D plot completed.')



if __name__ == "__main__":
    raise NotImplementedError("This script is not ready yet to run directly from here.")