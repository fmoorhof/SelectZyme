"""
This file provides basic functionalites like file parsing and protein large language model embedding. Additionally, it provides functions to create a vector database collection in Qdrant and to load it from there.
"""
import logging

from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient, models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_embedding(sequences: list[str], plm_model: str = 'esm1b', no_pad: bool = False) -> np.ndarray:
    """
    Generate embeddings for a list of sequences using a specified pre-trained language model (PLM).
    Args:
        sequences (list[str]): A list of sequences to generate embeddings for.
        plm_model (str, optional): The pre-trained language model to use. Default is 'esm1b'.
                                   Supported models include 'esm1b', 'prott5', and 'prostt5'.
        no_pad (bool, optional): If True, removes paddings from the sequences before generating mean embeddings.
                                 Default is False.
    Returns:
        np.ndarray: An array of embeddings for the input sequences.
    """

    tokenizer, model = _select_plm_model(plm_model)
    logging.info(f"Generating {plm_model} embeddings using device: {device}")
    
    if plm_model == 'prott5' or plm_model == 'prostt5':  # models require sequenes to be spaced (whitespace)
        sequences = [" ".join(list(seq)) for seq in sequences]
    
    embeddings = []
    for seq in tqdm(sequences):
        inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(**inputs)

        if no_pad == False:
            last_hidden_states = outputs.last_hidden_state
            embedding = torch.mean(last_hidden_states, dim=1).squeeze().cpu().numpy()  # Move embedding back to CPU
        else:
            if plm_model == 'prott5' or plm_model == 'prostt5':
                seq_len = int(len(seq)/2+1)
            else:
                seq_len = len(seq)
            last_hidden_states = outputs.last_hidden_state[0, :seq_len, :]  # Remove padding
            embedding = last_hidden_states.mean(dim=0).cpu().numpy()  # Move embedding back to CPU
        embeddings.append(embedding)

    return np.array(embeddings)


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
        id = i.payload.get('accession')
        id_embed[id] = vector
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


def _select_plm_model(plm_model: str = 'esm1b') -> tuple:
    """
    Selects and loads a pre-trained language model (PLM) and its corresponding tokenizer based on the specified model name.
    Args:
        plm_model (str): The name of the pre-trained language model to load. 
                         Options are 'esm1b' (default), 'esm2', 'esm3', 'prott5', 'prostt5'.
    Returns:
        tuple: A tuple containing the tokenizer and the model.
    Raises:
        ValueError: If the specified model name is not supported.
        NotImplementedError: If 'esm3' is specified, as it is not yet implemented.
    """

    if plm_model == 'esm1b':
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
        model = AutoModel.from_pretrained("facebook/esm1b_t33_650M_UR50S").to(device)

    elif plm_model == 'esm2':
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)

    elif plm_model == 'esm3':
        raise NotImplementedError("ESM3 not yet implemented")
        # tokenizer = AutoTokenizer.from_pretrained("EvolutionaryScale/esm3-sm-open-v1")  # no tokenizer specified!?!
        model = AutoModel.from_pretrained("EvolutionaryScale/esm3-sm-open-v1").to(device)

    elif plm_model == 'prott5':
        from transformers import T5Tokenizer, T5EncoderModel

        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)

    elif plm_model == 'prostt5':
        from transformers import T5Tokenizer, T5EncoderModel

        tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5")
        model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)

    else:
        raise ValueError(f"Model {model} not supported. Please choose one of: 'esm1b' (default), 'esm2', 'esm3', 'prott5', 'prostt5', 'saprot'")
    
    return tokenizer, model



if __name__=='__main__':
    # load example data
    from preprocessing import Parsing
    from preprocessing import Preprocessing

    df = Parsing.parse_tsv('tests/head_10.tsv')
    # df = Preprocessing(df).preprocess()
    pp = Preprocessing(df)
    # pp.remove_long_sequences()
    pp.remove_sequences_without_Metheonin()
    # pp.remove_sequences_with_undertermined_amino_acids()
    pp.remove_duplicate_entries()
    pp.remove_duplicate_sequences()    
    df = pp.df


    # test embedding
    embeddings = gen_embedding(df['sequence'].tolist(), plm_model='esm2')  # , no_pad=True)
    print(embeddings.shape)
    print(embeddings)

    # test vector db
    collection_name='pytest'
    # qdrant = QdrantClient(path="/data/tmp/EnzyNavi/")  # OR write them to disk
    # annotation, embeddings = load_or_createDB(qdrant, df, collection_name=collection_name, plm_model='esm2')