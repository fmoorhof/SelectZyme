"""This file should assist in loading .h5 files from the uniprot embeddings project. The script has mainly the functionality of embed.py but
is acting on 5h files and not on a dataframe. That is why i want to keep it seperate from embed.py.

vectors are 1024-dimensional float16 vectors, e.g.:
array([], dtype=float16)
"""
import logging

import h5py  # todo: not yet appended to requirements.txt
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from embed import load_collection_from_vector_db


def create_db_from_5h(filename: str, collection_name: str) -> list:
    """
    Create a vector database from a .h5 file containing the uniprot embeddings.

    :param filename: path to the .h5 file
    :param collection_name: name of the vector database
    return: annotation: list of 'Entry'
    return: list of vectors (data matrix X)
    """
    # Load the h5 data
    f = h5py.File(filename, 'r')

    entries = list(f.keys())
    # entries = entries[0:6]  # small test dataset for debugging (same size like head_10.tsv after preprocessing)
    vector_size = f[entries[0]].shape[0]
    logging.info(f"The vectors are of dimension: {vector_size}")
    logging.info(f"Got {len(entries)} entries from {filename}.")

    qdrant = QdrantClient(path="/scratch/global_1/fmoorhof/Databases/Vector_db/")  # host="http://localhost:6333")
    collections_info = qdrant.get_collections()
    collection_names = [collection.name for collection in collections_info.collections]
    if collection_name not in collection_names:
        logging.info(f"Vector DB doesnt exist yet. A Qdrant vector DB will be created under path=/scratch/global_1/fmoorhof/Databases/Vector_db/")
        qdrant.create_collection(collection_name=collection_name, vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            ))
    else:  # todo: define loading instead, and make use of embed.py! to recycle code
        logging.info(f"Vector DB collection exists already and will be read.")
        annotations, X = load_collection_from_vector_db(qdrant, collection_name=collection_name)
        return annotations, X

    X = []
    annotations = []
    records = []
    logging.info(f"Creating Qdrant records. This may take a while.")
    for i, entry in enumerate(tqdm(entries)):  # see embed.create_vector_db_collection() for an alternative implementation
        vector = f[entry][:].tolist()
        annotation = f.get(entry).attrs["original_id"]
        record = models.Record(id=i, vector=vector, payload={'Entry': entry})  # payload needs to be a dict
        records.append(record)
        annotations.append(annotation)

    logging.info(f"Uploading data to Qdrant DB. This may take a while.")
    qdrant.upload_records(
        collection_name=collection_name,
        records=records
    )

    return annotations, X



if __name__ == '__main__':
    filename = '/scratch/global_1/fmoorhof/Databases/per-protein.h5'
    collection_name='swiss-prot2024-01-14_testing'
    create_db_from_5h(filename, collection_name)