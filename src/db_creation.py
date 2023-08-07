"""
This file is creating a Qdrant vector database for protein sequences. Within this file the .fasta gets parsed, embedded and saved.

Outlook:
It is possible to have multiple vectors per record. This feature allows for multiple vector storages per collection
-> append info to a seq. from other LLMs
"""
import logging

# downstream analysis: not particularly needed here
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from minimal import gen_embedding, read_fasta
from qdrant_client import QdrantClient, models  # ! pip install qdrant-client
from scipy.spatial.distance import cdist
from umap import UMAP

logging.basicConfig(
    format="%(levelname)-8s| %(module)s.%(funcName)s: %(message)s", level=logging.DEBUG
)


def filter_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function removes too long sequences from the dataset. Sequences > 1024 amino acids cause the esm embedding to fail.
    """
    old_len = df.shape[0]
    df = df[df["Sequence"].str.len() <= 1024]
    df.drop_duplicates(inplace=True)
    df.reset_index(
        drop=True, inplace=True
    )  # todo: with removed index log which sequences were excluded
    logging.info(
        f"{old_len-df.shape[0]} sequences were excluded because of exaggerated size (>=1024 amino acids)"
    )

    return df


# qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance
qdrant = QdrantClient(path="Vector_db/")  # Persists changes to disk
collection_name = "test_parsing_uniprot"
collection_name = "test_parsing_batch2_annotated_headtail"
# collection_name = '2ogd_ec_50'
# collection_name = '2ogd_ec_only'
# collection_name = '2ogd_full'
in_data = "datasets/ec_only_clustered50.fasta"
in_data = "datasets/test_parsing_uniprot.tsv"
# in_data = 'datasets/ec_only_annotated.tsv'
# in_data = 'datasets/batch2_annotated.tsv'
in_data = "datasets/batch2_annotated_headtail.tsv"


# Check if the collection exists yet
collections_info = qdrant.get_collections()
if collection_name not in str(
    collections_info
):  # todo: implement this nicely: access the 'name' field of the object
    logging.info(
        f"Vector DB doesnt exist yet. Your .fasta will be embedded and a vector DB created under path=Vector_db/"
    )

    # todo: not tested yet the .fasta parsing!
    # read sequences either from .fasta or .tsv
    if ".fasta" in in_data:
        annotation = []
        sequences = []
        for h, s in read_fasta(in_data):
            annotation.append(h)
            sequences.append(s)
        df = pd.DataFrame(annotation)
        df["Sequence"] = sequences
        df = filter_sequences(df=df)
        logging.info(f"Read from .fasta file: {in_data}'")

    else:
        # read from .tsv
        df = pd.read_csv(in_data, delimiter="\t")
        df = filter_sequences(df=df)
        logging.info(f"Read from .tsv file: {in_data}'")

    # embed
    embeddings = gen_embedding(
        df["Sequence"].tolist(), device="cuda"
    )  # fails if sequences >= 1024 aa
    logging.info(f"The embeddings have the dimension: '{embeddings.shape}'")

    # OR Create collection to store sequences
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embeddings.shape[1],  # Vector size is defined by used model
            distance=models.Distance.COSINE,
        ),
    )

    # Let's vectorize descriptions and upload to qdrant
    # header = [{'name': x} for x in headers]  # dict conversion (need as parameter payload=)
    # header_idx_dict = {heads:idx for idx, heads in enumerate(headers)}  # dict comprehension  # todo: adapt this if header != UniProtID only
    annotation = df.iloc[:, :-1].to_dict(
        orient="index"
    )  # headers = df.loc[:, 'Entry':'Length'].T.to_dict()
    qdrant.upload_records(
        collection_name=collection_name,
        records=[  # strange structure requirements
            models.Record(
                id=idx,
                vector=embeddings[
                    idx
                ].tolist(),  # each embedding needs to be a python list, not np.array
                payload=heads  # dict/json required; payload is ability to store additional information along with vectors
                # ) for idx, heads in enumerate(header)
            )
            for idx, heads in annotation.items()
        ],
    )


# Retrieve all points of a collection with defined return fields (payload e.g.)
# A point is a record consisting of a vector and an optional payload
logging.info(
    f"Retrieving data from Qdrant vector DB. This may take a while for some 100k sequences."
)
collection = qdrant.get_collection(collection_name)
records = qdrant.scroll(
    collection_name=collection_name,
    with_payload=True,  # If List of string - include only specified fields
    with_vectors=True,
    limit=collection.vectors_count,
)  # Tuple(Records, size)
# qdrant.delete_collection(collection_name)

# extract the header and vector from the Qdrant data structure
id_embed = {}
annotation = []
for i in records[0]:  # access only the Records: [0]
    vector = i.vector
    id = i.payload.get("Entry")
    id_embed[id] = vector
    annotation.append(i.payload)
embeddings = np.array(
    list(id_embed.values())
)  # dimension error if dataset has duplicates
df = pd.DataFrame(annotation)


# Visualization: Dimensionality reduction
# todo: outsource this part into real analysis; todo: annotate plot with EC number
logging.info(
    f"Starting 2D UMAP projection. This may take a while for some 100k sequences."
)
umap = UMAP(
    n_components=2,
    densmap=True,
    metric="precomputed",
    output_metric="euclidean",
    random_state=42,
)
# distmat = Metrics.cosine(embeddings, embeddings)
distmat = cdist(embeddings, embeddings, metric="cosine")
_viz = umap.fit_transform(distmat).astype(np.float16)  # slow for many seqs

# Visualize the clusters using Plotly Express
# flatten = np.array([np.array(i).flatten() for i in _viz])
df["x"] = _viz[:, 0]  # todo: buggs here if .fasta only has Accession as solely header
df["y"] = _viz[:, 1]
# todo: this part fails if the user doesnt have these fields. Make change optional
# df gets changed but the plotly visualization strangely not. dont know what the problem is there
# df.loc[df['Reviewed'].str.contains('unreviewed'), 'shape_property'] = 'circle'
# df.loc[~df['Reviewed'].str.contains('unreviewed'), 'shape_property'] = 'diamond'
df.loc[
    df["EC number"].str.contains("    "), "EC number"
] = "0.0.0.0"  # convert empty ECs to an artificial new EC class
cols = df.columns.values.tolist()
cols = cols[0:-3]
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="EC number",
    title="UMAP in 2D",
    hover_data=cols,
    color_continuous_scale=px.colors.qualitative.Plotly,
    # symbol='shape_property',
    opacity=0.8,
)
# fig.write_html(f"Output/{collection_name}_umap.html")


import plotly.graph_objects as go

# Simple Clustering with different plotting (EC numbers can be resolved with this implementation. however not able to show cluster coloring yet :/)
from sklearn.cluster import DBSCAN

X = embeddings
logging.info(f"Starting DBSCAN. This may take a while for some 100k sequences.")
dbscan = DBSCAN(eps=1.0, min_samples=1)
labels = dbscan.fit_predict(X)
df["cluster"] = labels

# Create the scatter plot with hovertext
fig = go.Figure(
    data=go.Scatter(
        x=df["x"],
        y=df["y"],
        mode="markers",
        hovertext=df["EC number"],  # Specify the column to show in hovertext
        marker=dict(size=10),
    )
)
# fig.write_html(f"Output/{collection_name}_umap_dbscan.html")


# cluster statistics
cluster_counts = df["cluster"].value_counts().reset_index()
cluster_counts.columns = ["cluster", "count"]
# todo: remove cluster < 5 Sequences
# todo: dont plot arbitrary sum

# Create the histogram using Plotly Express
fig = px.histogram(
    cluster_counts,
    x="cluster",
    y="count",
    title="Cluster Frequencies",
    labels={"cluster": "Cluster", "count": "Frequency"},
    barmode="overlay",
)  # Set barmode to 'overlay' or 'group')
fig.write_html(f"Output/{collection_name}_histogram.html")


"""
# GridSearch Clustering
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.metrics import make_scorer

from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd

# todo: dead end: scoring function for un-labeled data / unsupervised learning not existent

# Perform GridSearch to find the best hyperparameterscollection
# initial/random hyperparameters are awful. We need a metric and hyperparameter optimization -> silhouette_score
param_grid = {
    'eps': np.linspace(0.1, 1.0, 4),
    'min_samples': [2, 10, 20]
}
# Define the silhouette score as the evaluation metric
silhouette_scorer = make_scorer(silhouette_score)
dbscan = DBSCAN()
X = embeddings

# Perform GridSearchCV to find the best combination of hyperparameters
grid_search = GridSearchCV(dbscan, param_grid, scoring=silhouette_scorer, cv=5)  # scoring
grid_search.fit(X)

# Get the best hyperparameters
best_eps = grid_search.best_params_['eps']
best_min_samples = grid_search.best_params_['min_samples']

# Create the final DBSCAN model with the best hyperparameters
dbscan_final = DBSCAN(eps=best_eps, min_samples=best_min_samples)
labels = dbscan_final.fit_predict(X)


# Dim reduced visualization with PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Create a DataFrame with the 2D data and cluster labels
df = pd.DataFrame({'x': X_2d[:, 0], 'y': X_2d[:, 1], 'cluster': labels})

# Visualize the clusters using Plotly Express
fig = px.scatter(df, x='x', y='y', color='cluster',
                 title='DBSCAN Clustering in 2D', 
                 color_continuous_scale=px.colors.qualitative.Plotly)
fig.write_html("Output/{collection_name}_dbscan_pca.html")
"""


"""
# Let's now search for something
# encode the search
query = gen_embedding(sequences[:1], device='cuda')  # todo: bugfix sequences[:] not available if db file used
# search
hits = qdrant.search(
    collection_name=collection_name,
    query_vector=query[0],  # accepts only 1D list
    limit=3
)
for hit in hits:
  print(hit.payload, "score:", hit.score)  # todo: fix: why is payload empty
# todo: try search_groups -> implementation to search with entire lists

# Let's now search only for books from 21st century
# # -> query with conditions  

"""
