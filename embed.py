"""
This file provides basic functionalites like file parsing and esm embedding.
"""
import logging

import pandas as pd
import numpy as np
import torch
import esm
from qdrant_client import QdrantClient, models  # ! pip install qdrant-client

# testing
from preprocessing import Parsing


from preprocessing import Preprocessing


def gen_embedding(sequences, device: str = 'cuda'):
    """
    Generate embeddings for a list of protein sequences.

    :param sequences: list containing protein sequences to embed here
    :param device: device for running the model (either cpu or gpu=cuda)
    """
    # load the esm-1b protein language model
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    model.eval()  # disable dropout for deterministic results
    model = model.to(device)

    embeddings = []

    with torch.no_grad():
        for n, s in enumerate(sequences):
            logging.info(f'Progress : {n+1} / {len(sequences)}\r')
            
            batch_labels, batch_strs, batch_tokens = batch_converter([[None, s]])
            batch_tokens = batch_tokens.to(device)
            
            # generate the full size embedding vector
            result = model(batch_tokens, repr_layers=[33], return_contacts=False)
            full_size = result["representations"][33].to('cpu')[0]
            
            # derive a fixed size embedding vector
            fixed_size = full_size[1:-1].mean(0).numpy()  # other than mean possible, too
            embeddings.append(fixed_size)
    embeddings = np.array(embeddings)
    logging.info(f"The embeddings have the dimension: '{embeddings.shape}'")

    return embeddings


def create_vector_db_collection(qdrant, df, embeddings, collection_name: str):
    """
    Create a vector database with the embeddings of the sequences and the annotation from the dataframe (but not the sequences themselves).
    The DB will be created at: "datasets/Vector_db/"

    :param df: dataframe containing the sequences and the annotation
    :param embeddings: numpy array containing the embeddings
    :param collection_name: name of the vector database
    """
    logging.info(f"Vector DB doesnt exist yet. A Qdrant vector DB will be created under path=Vector_db/")

    # OR Create collection to store sequences
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embeddings.shape[1], # Vector size is defined by used model
            distance=models.Distance.COSINE
        )
        )

    # Vectorize descriptions and upload to qdrant
    # header = [{'name': x} for x in headers]  # dict conversion (need as parameter payload=)
    # header_idx_dict = {heads:idx for idx, heads in enumerate(headers)}  # dict comprehension  # todo: adapt this if header != UniProtID only
    annotation = df.iloc[:, :-1].to_dict(orient='index')  # headers = df.loc[:, 'Entry':'Length'].T.to_dict()
    qdrant.upload_records(
        collection_name=collection_name,
        records=[  # strange structure requirements
            models.Record(
                id=idx,
                # todo: fix: ValueError: Out of range float values are not JSON compliant
                vector=embeddings[idx].tolist(),   # each embedding needs to be a python list, not np.array
                payload=heads  # dict/json required; payload is ability to store additional information along with vectors
            # ) for idx, heads in enumerate(header)
            ) for idx, heads in annotation.items()
        ]
    )
    df = pd.DataFrame(annotation)
    return df, embeddings


def load_collection_from_vector_db(collection_name: str):
    """xy"""
    logging.info(f"Retrieving data from Qdrant DB. This may take a while for some 100k sequences.")
    # Retrieve all points of a collection with defined return fields (payload e.g.)
    # A point is a record consisting of a vector and an optional payload
    collection = qdrant.get_collection(collection_name)
    records = qdrant.scroll(collection_name=collection_name,
                            with_payload=True,  # If List of string - include only specified fields
                            with_vectors=True,
                            limit=collection.vectors_count)  # Tuple(Records, size)
    # qdrant.delete_collection(collection_name)

    # extract the header and vector from the Qdrant data structure
    id_embed = {}
    annotation = []
    for i in records[0]:  # access only the Records: [0]
        vector = i.vector
        id = i.payload.get('Entry')
        id_embed[id] = vector
        annotation.append(i.payload)
    embeddings = np.array(list(id_embed.values()))  # dimension error if dataset has duplicates
    df = pd.DataFrame(annotation)
    return df, embeddings

    

if __name__=='__main__':

    df = Parsing.parse_tsv('tests/head_10.tsv')
    pp = Preprocessing(df)
    df = pp.remove_long_sequenes()
    collection_name='pytest'

    embeddings = gen_embedding(df['Sequence'].tolist(), device='cuda')

    # Check if the collection exists yet if not create it
    # qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance
    qdrant = QdrantClient(path="datasets/Vector_db/")  # OR write them to disk
    collections_info = qdrant.get_collections()
    if collection_name not in str(collections_info):  # todo: implement this nicely: access the 'name' field of the object
        create_vector_db_collection(qdrant, df, embeddings, collection_name='test_collection')
        print('created collection')
    else:
        df = load_collection_from_vector_db(qdrant, collection_name='test_collection')
        print('loaded collection')