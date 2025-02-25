from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from src.embed import gen_embedding
from src.utils import run_time


@run_time
def database_access(df: pd.DataFrame, collection_name: str, plm_model: str = "esm1b") -> np.ndarray:
    """Create a collection in Qdrant DB with embedded sequences
    :param df: dataframe containing the sequences and the annotation
    :param collection_name: name of the collection
    return: embeddings: numpy array containing the embeddings
    
    Additional remarks:
    Start the docker container for qdrant server e.g. with:
    docker run -p 6333:6333 -p 6334:6334 -v "/data/tmp/EnzyNavi/qdrant_storage:/qdrant/storage:z" fmoorhof/qdrant:1.13.2

    -p expose ports, -v mount volume
    """
    logging.info("Instantiating Qdrant vector DB. This takes quite a while when its not continously running as container.")
    qdrant = QdrantClient(
        url="http://localhost:6333", timeout=15
    )
    
    # Check if collection already exists
    collections_info = qdrant.get_collections()
    collection_names = [collection.name for collection in collections_info.collections]
    if collection_name not in collection_names:  # create it
        embeddings = gen_embedding(df["sequence"].tolist(), plm_model=plm_model)
        create_collection(qdrant, embeddings, collection_name=collection_name)
        upload_points(qdrant, embeddings, collection_name=collection_name)

        if (df.shape[0] != embeddings.shape[0]):
            raise ValueError(
                f"Length of dataframe ({df.shape[0]}) and embeddings ({embeddings.shape[0]}) do not match. \
                Something went wrong, you might have duplicate entries (accession) in your dataset. \
                Accession must be unique")
            
    else:  # load it
        embeddings = load_collection_from_vector_db(qdrant, collection_name)

        if (df.shape[0] != embeddings.shape[0]):
            logging.info(
                f"Length of dataframe ({df.shape[0]}) and embeddings ({embeddings.shape[0]}) do not match. \
                As a consequence the collection is replaced by your new one. \
                Use other 'project name' in the configuration to not overwrite collections."
            )
            qdrant.delete_collection(collection_name)
            embeddings = gen_embedding(df["sequence"].tolist(), plm_model=plm_model)
            create_collection(qdrant, embeddings, collection_name=collection_name)
            upload_points(qdrant, embeddings, collection_name=collection_name)

    # todo: is this really needed?
    # sys.setrecursionlimit(
    #     max(df.shape[0], 10000)
    # )  # fixed: RecursionError: maximum recursion depth exceeded

    return embeddings


def create_collection(qdrant, embeddings: np.ndarray, collection_name: str) -> None:
    logging.info(
        "Vector DB doesnt exist yet. A Qdrant vector DB will be created under path=Vector_db/"
    )
    # Create empty collection to store sequences
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embeddings.shape[1],  # Vector size is defined by used model
            distance=models.Distance.EUCLID,
        ),
    )


def upload_points(qdrant, embeddings, collection_name: str) -> None:
    """
    Create a vector database with the embeddings of the sequences and the annotation from the dataframe (but not the sequences themselves).

    :param df: dataframe containing the sequences and the annotation
    :param embeddings: numpy array containing the embeddings
    :param collection_name: name of the vector database
    """
    records = []
    for i, embedding in enumerate(embeddings):
        record = models.Record(id=i, vector=embedding.tolist())
        records.append(record)
    qdrant.upload_records(collection_name=collection_name, records=records)


def load_collection_from_vector_db(qdrant, collection_name: str) -> list:
    """
    Load the collection from the vector database.
    # Retrieve all points of a collection with defined return fields (payload e.g.)
    # A point is a record consisting of a vector and an optional payload

    :param qdrant: qdrant object
    :param collection_name: name of the vector database
    return: embeddings: numpy array containing the embeddings"""
    collection = qdrant.get_collection(collection_name)
    records = qdrant.scroll(
        collection_name=collection_name,
        with_vectors=True,
        limit=collection.points_count,
        timeout=190,
    )

    # extract the vectors from the Qdrant records
    embeddings = []
    for i in tqdm(records[0]):  # access only the Records: [0]
        embeddings.append(i.vector)

    return np.array(embeddings)
