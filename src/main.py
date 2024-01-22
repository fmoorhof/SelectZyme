import logging

import pandas as pd
from qdrant_client import QdrantClient, models


from preprocessing import Parsing
from preprocessing import Preprocessing
from embed import *


logging.basicConfig(
    format="%(levelname)-8s| %(module)s.%(funcName)s: %(message)s", level=logging.DEBUG
)


def db_creation(df, collection_name: str):
    collections_info = qdrant.get_collections()
    if collection_name not in str(collections_info):  # todo: implement this nicely: access the 'name' field of the object
        embeddings = gen_embedding(df['Sequence'].tolist(), device='cuda')
        create_vector_db_collection(df, embeddings, collection_name=collection_name)
    else:
        df = load_collection_from_vector_db(collection_name='test_collection')


def main(input_file: str, project_name: str):

    if input_file.endswith('.fasta'):
        headers, sequences = Parsing.parse_fasta(input_file)
        df = pd.DataFrame({'Header': headers, 'Sequence': sequences})
    else:
        df = Parsing.parse_tsv(input_file)

    # df needs to contain a column 'Sequence' with the sequences
    pp = Preprocessing(df)
    df = pp.remove_long_sequenes(df)
    df = pp.remove_sequences_without_Metheonin(df)
    df = pp.remove_sequences_with_undertermined_amino_acids(df)

    # Create a collection in Qdrant DB with embedded sequences
    db_creation(df, collection_name=project_name)



if __name__ == "__main__":
    main(input_file='tests/head_10.tsv', project_name='test_project')
