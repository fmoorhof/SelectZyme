"""This file should assist in loading .h5 files from the uniprot embeddings project.

vectors are 1024-dimensional float16 vectors, e.g.:
array([], dtype=float16)
"""
import logging

import h5py  # todo: not yet appended to requirements.txt
from qdrant_client import QdrantClient, models


def create_db_from_5h(filename: str, collection_name: str) -> list:
    """
    Create a vector database from a .h5 file containing the uniprot embeddings.

    :param filename: path to the .h5 file
    :param collection_name: name of the vector database
    return: list of vectors (data matrix X)
    """
    # Load the h5 data
    f = h5py.File(filename, 'r')

    entries = list(f.keys())
    vector_size = f[entries[0]].shape[0]
    logging.info(f"The vectors are of dimension: {vector_size}")
    logging.info(f"Got {len(entries)} entries from {filename}.")

    qdrant = QdrantClient(path="/scratch/global_1/fmoorhof/Databases/Vector_db/")  # host="http://localhost:6333")
    qdrant.create_collection(collection_name=collection_name, vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            ))

    X = []
    records = []
    for i, entry in enumerate(entries):  # see embed.create_vector_db_collection() for an alternative implementation
        vector = f[entry][:].tolist()
        annotation = f.get(entry).attrs["original_id"]
        record = models.Record(id=i, vector=vector, payload={entry: entry})  # payload needs to be a dict
        records.append(record)

    qdrant.upload_records(
        collection_name=collection_name,
        records=records
    )

    return X



if __name__ == '__main__':
    filename = '/scratch/global_1/fmoorhof/Databases/per-protein.h5'
    collection_name='swiss-prot2024-01-14'
    create_db_from_5h(filename, collection_name)