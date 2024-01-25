"""
This file provides basic functionalites like file parsing and esm embedding.
"""
import logging

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import esm
from qdrant_client import QdrantClient, models  # ! pip install qdrant-client

# testing
from preprocessing import Parsing


from preprocessing import Preprocessing


def gen_embedding(sequences, device: str = 'cuda:0'):
    """
    Generate embeddings for a list of protein sequences.

    :param sequences: list containing protein sequences to embed here
    :param device: device for running the model (either cpu or gpu=cuda), :number specifies the gpu (if you have multiple use >1 e.g. cuda:1)
    """
    # load the esm-1b protein language model
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    model.eval()  # disable dropout for deterministic results
    model = model.to(device)

    embeddings = []

    with torch.no_grad():
        for sequence in tqdm(sequences):
            batch_labels, batch_strs, batch_tokens = batch_converter([[None, sequence]])
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


def create_vector_db_collection(qdrant, df, embeddings, collection_name: str) -> list:
    """
    Create a vector database with the embeddings of the sequences and the annotation from the dataframe (but not the sequences themselves).

    :param df: dataframe containing the sequences and the annotation
    :param embeddings: numpy array containing the embeddings
    :param collection_name: name of the vector database
    return: annotation: list of 'Entry'
    """
    logging.info(f"Vector DB doesnt exist yet. A Qdrant vector DB will be created under path=Vector_db/")

    # Create empty collection to store sequences
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embeddings.shape[1], # Vector size is defined by used model
            distance=models.Distance.COSINE
        )
        )
    
    records = []
    annotation = df.iloc[:, 0:2].to_dict(orient='index')  # only use 'Entry' as key (0:1)  # (0, {'Entry': 'Q9NWT6', 'Reviewed': True})
    for i, anno in annotation.items():
        vector = embeddings[i].tolist()
        record = models.Record(id=i, vector=vector, payload=anno)  # {'Entry': 'Q9NWT6', 'Reviewed': True}
        records.append(record)

    # Upload records to Qdrant collection
    qdrant.upload_points(
        collection_name=collection_name,
        records=records
    )
    
    return annotation


def load_collection_from_vector_db(qdrant, collection_name: str) -> list:
    """
    Load the collection from the vector database. 
    # Retrieve all points of a collection with defined return fields (payload e.g.)
    # A point is a record consisting of a vector and an optional payload

    :param qdrant: qdrant object
    :param collection_name: name of the vector database
    return: annotation: list of 'Entry'
    return: embeddings: numpy array containing the embeddings"""
    logging.info(f"Retrieving data from Qdrant DB. This may take a while for some 100k sequences.")
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
        # annotation.append(i.payload)  # theoretically more information than only 'Entry' could be stored/retrieved
        annotation.append(i.payload['Entry'])
    embeddings = np.array(list(id_embed.values()))  # dimension error if dataset has duplicates
    # df = pd.DataFrame(annotation)
    return annotation, embeddings

    

if __name__=='__main__':

    df = Parsing.parse_tsv('tests/head_10.tsv')
    pp = Preprocessing(df)
    pp.remove_long_sequences()
    pp.remove_sequences_without_Metheonin()
    pp.remove_sequences_with_undertermined_amino_acids()
    df = pp.df
    collection_name='pytest'

    embeddings = gen_embedding(df['Sequence'].tolist(), device='cuda:1')

    # Check if the collection exists yet if not create it
    # qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance
    qdrant = QdrantClient(path="datasets/Vector_db/")  # OR write them to disk

    qdrant.delete_collection(collection_name)  # todo: remove this after code development

    collections_info = qdrant.get_collections()

    if collection_name not in str(collections_info):  # todo: implement this nicely: access the 'name' field of the object
        create_vector_db_collection(qdrant, df, embeddings, collection_name)
        print('created collection')
        collection = qdrant.get_collection(collection_name)
        records = qdrant.scroll(collection_name=collection_name,
                            with_payload=True,  # If List of string - include only specified fields
                            with_vectors=True,
                            limit=collection.vectors_count)
        print(records[0][1].payload) 
        annotation, embeddings = load_collection_from_vector_db(qdrant, collection_name)
        print(annotation, embeddings)
    else:
        annotation, embeddings = load_collection_from_vector_db(qdrant, collection_name)
        print(annotation, embeddings)