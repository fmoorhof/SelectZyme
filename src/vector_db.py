import logging

from tqdm import tqdm
import numpy as np
from qdrant_client import QdrantClient, models

from src.embed import gen_embedding


def create_vector_db_collection(qdrant, df, embeddings, collection_name: str) -> list:
    """
    Create a vector database with the embeddings of the sequences and the annotation from the dataframe (but not the sequences themselves).

    :param df: dataframe containing the sequences and the annotation
    :param embeddings: numpy array containing the embeddings
    :param collection_name: name of the vector database
    return: annotation: list of 'accession'
    """
    logging.info("Vector DB doesnt exist yet. A Qdrant vector DB will be created under path=Vector_db/")

    # Create empty collection to store sequences
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embeddings.shape[1], # Vector size is defined by used model
            distance=models.Distance.COSINE
        )
        )
    
    records = []
    annotation = df.iloc[:, 0:2].to_dict(orient='index')  # only use 'accession' as key (0:1)  # (0, {'accession': 'Q9NWT6', 'Reviewed': True})
    logging.info("Creating Qdrant records. This may take a while.")
    for i, anno in tqdm(annotation.items()):
        vector = embeddings[i].tolist()
        record = models.Record(id=i, vector=vector, payload=anno)  # {'accession': 'Q9NWT6', 'Reviewed': True}
        records.append(record)

    logging.info("Uploading data to Qdrant DB. This may take a while.")
    qdrant.upload_records(
        collection_name=collection_name,
        records=records
    )
    
    return annotation, embeddings


def load_collection_from_vector_db(qdrant, collection_name: str) -> list:
    """
    Load the collection from the vector database. 
    # Retrieve all points of a collection with defined return fields (payload e.g.)
    # A point is a record consisting of a vector and an optional payload

    :param qdrant: qdrant object
    :param collection_name: name of the vector database
    return: annotation: list of 'accession'
    return: embeddings: numpy array containing the embeddings"""
    collection = qdrant.get_collection(collection_name)
    records = qdrant.scroll(collection_name=collection_name,
                            with_payload=True,  # If List of string - include only specified fields
                            with_vectors=True,
                            limit=collection.vectors_count)  # Tuple(Records, size)
    # qdrant.delete_collection(collection_name)

    # extract the header and vector from the Qdrant data structure
    id_embed = {}
    annotation = []
    for i in tqdm(records[0]):  # access only the Records: [0]
        vector = i.vector
        idd = i.payload.get('accession')
        id_embed[idd] = vector
        # annotation.append(i.payload)  # theoretically more information than only 'Entry' could be stored/retrieved
        annotation.append(i.payload['accession'])
    embeddings = np.array(list(id_embed.values()))  # dimension error if dataset has duplicates
    # df = pd.DataFrame(annotation)
    return annotation, embeddings

    
def load_or_createDB(qdrant: QdrantClient, df, collection_name: str, plm_model: str = 'esm1b'):
    """Checks if a collection with the given name already exists. If not, it will be created.
    :param qdrant: qdrant object
    :param df: dataframe containing the sequences and the annotation
    :param collection_name: name of the vector database
    return: annotation: list of 'Entry'
    return: embeddings: numpy array containing the embeddings"""
    # qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance
    collections_info = qdrant.get_collections()
    collection_names = [collection.name for collection in collections_info.collections]
    if collection_name not in collection_names:
        embeddings = gen_embedding(df['sequence'].tolist(), plm_model=plm_model)
        annotation, embeddings = create_vector_db_collection(qdrant, df, embeddings, collection_name=collection_name)
    else:
        annotation, embeddings = load_collection_from_vector_db(qdrant, collection_name)
    return annotation, embeddings