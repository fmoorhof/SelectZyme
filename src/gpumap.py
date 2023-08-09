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
import sys

import cudf
import matplotlib.pyplot as plt

# downstram analysis: not particularly needed here
import plotly.express as px
import plotly.figure_factory as ff
from cuml.cluster import (
    HDBSCAN,
)  # pip install hdbscan (the cuml is based on it else plotting can not be done direcly from the module)
from cuml.cluster import DBSCAN, AgglomerativeClustering
from cuml.decomposition import PCA

# ! pip install qdrant-client
from qdrant_client import QdrantClient, models

# qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance
qdrant = QdrantClient(path="Vector_db/")  # Persists changes to disk
collection_name = "2ogd_ec_50"
collection_name = "2ogd_ec_only"
# collection_name = '2ogd_full'
# collection_name = 'test_parsing'
# collection_name = 'test_parsing_batch2_annotated_headtail'


# Retrieve all points of a collection with defined return fields (payload e.g.)  # takes ~10 mins for 200k
# A point is a record consisting of a vector and an optional payload
collection = qdrant.get_collection(collection_name)
records = qdrant.scroll(
    collection_name=collection_name,
    with_payload=True,  # If List of string - include only specified fields
    with_vectors=True,
    limit=collection.vectors_count,
)  # Tuple(Records, size)
# qdrant.delete_collection(collection_name)
logging.info(f"Collection {collection_name} loaded")

# extract the header and vector from the Qdrant data structure
id_embed = {}
annotation = []
for i in records[0]:  # access only the Records: [0]
    vector = i.vector
    id = i.payload.get("Entry")
    id_embed[id] = vector
    annotation.append(i.payload)
embeddings = cudf.DataFrame(
    list(id_embed.values())
)  # dimension error if dataset has duplicates
df = cudf.DataFrame(annotation)

X = embeddings


# If your dataset is too large, you need to increase the recursion depth for the hierarchical clustering
sys.setrecursionlimit(203252)

# Clustering  # finished in 12 mins on 200k:)
min_samples = 30  # 30 worked good for ec_only; 50 for 200k
hdbscan = HDBSCAN(
    min_samples=min_samples, gen_min_span_tree=True
)  # min_samples= amount of how many points shall be in a neighborhood of a point to form a cluster
labels = hdbscan.fit_predict(X)
logging.info(f"HDBSCAN done")

# insert here the plottings again


# Clustering
# dbscan = DBSCAN(eps=1.0, min_samples=1)
# labels = dbscan.fit_predict(X)
logging.info(f"DBSCAN done")


# Dim reduced visualization with PCA
pca = PCA(n_components=3, output_type="numpy")
X_pca = pca.fit_transform(X)
variance = pca.explained_variance_ratio_ * 100
variance = ["%.1f" % i for i in variance]  # 1 decimal only
print(f"% Variance of the PCA components: {variance}")
logging.info(f"PCA done")

# Visualize the clusters using Plotly Express
df["cluster"] = labels
df["x"] = X_pca[:, 0]
df["y"] = X_pca[:, 1]
# df['z'] = X_pca[:, 2]  # 2D looks sophisticated
# df['EC number'] = df['EC number'].fillna('0.0.0.0')  # replace empty ECs because they will not get plottet (if color='EC number')
cols = df.columns.values.tolist()
cols = cols[0:-2]
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="EC number",  # color='cluster'  # scatter_3d(, z='z')
    title=f"CuML HDBSCAN (min_samples = {min_samples}) with 2D-PCA (variance: {variance}) on dataset: {collection_name}",
    hover_data=cols,
    opacity=0.5,
    color_continuous_scale=px.colors.sequential.Viridis,  # _r = reversed  # color_discrete_sequence=px.colors.sequential.Viridis,
    # symbol=['diamond' if value == True else 'circle' for value in df['Reviewed'].to_pandas()],  # .to_pandas() needed in cudf
)
fig.write_html(f"Output/trash2_Viridis_diamond{collection_name}_hdbscan_pca.html")
logging.info(f"Script completed without errors")


# plotting the results
print(labels)
# Bug: dtype: int32 - Segmentation fault (core dumped)

hdbscan.single_linkage_tree_.plot()
plt.savefig(f"Output/{collection_name}_single_linkage_tree.pdf")
hdbscan.minimum_spanning_tree_.plot()
plt.savefig(f"Output/{collection_name}_minimum_spanning_tree.pdf")
hdbscan.condensed_tree.plot()
plt.savefig(f"Output/{collection_name}_condensed_tree.pdf")


"""
# single-linkage agglomerative clustering using the nearest neighbors (knn)
agglomerative_cluster = AgglomerativeClustering(n_clusters=5, affinity='cosine', linkage='single', connectivity='knn', n_neighbors=1023)  # linkage='ward'
agglomerative_cluster.fit(X)
# Get the cluster assignments for each data point
labels = agglomerative_cluster.labels_
logging.info(f'Agglomerative clustering done')
print(labels)

# Get the linkage matrix from the agglomerative clustering model
linkage_matrix = agglomerative_cluster.children_
# Convert the linkage matrix to a pandas DataFrame
linkage_df = cudf.DataFrame(linkage_matrix, columns=['left_child', 'right_child', 'distance', 'cluster_size'])  # , dtype=np.float32

print(linkage_df)
print(linkage_matrix)


# Create the dendrogram using plotly.figure_factory.create_dendrogram
fig = ff.create_dendrogram(linkage_matrix.to_numpy())
# Update layout for better visibility
fig.update_layout(width=600, height=400, title='Agglomerative Clustering Dendrogram', xaxis_title='Samples', yaxis_title='Distance')
fig.write_html(f"Output/{collection_name}_trash_singleLinkage.html")
logging.info(f'Script completed without errors')



"""
