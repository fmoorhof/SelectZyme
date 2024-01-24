import pytest
from qdrant_client import QdrantClient

import embed
from preprocessing import Parsing
from preprocessing import Preprocessing


class TestEmbedding:
    """Test the embedding function."""
    def setup_method(self, method):
        # Replace this with the code to create your DataFrame
        self.df = Parsing.parse_tsv('tests/head_10.tsv')
    
    sequences = ['MSLDTIPVVDLGPLLTGD', 'MSLDTIPVVDLGPLLTGD']

    @pytest.mark.parametrize("sequences", sequences)
    def test_embeddings(self, sequences): 
        """Test esm-1b embedding"""
        seqs = sequences
        embeddings = embed.gen_embedding(seqs, device='cuda')
        
        assert embeddings is not []
        assert embeddings.shape == (18, 1280)  # each of the 18 amino acids is embedded into a 1280 dimensional vector

    def test_gen_embedding(self):
        """Test esm-1b embedding with a list of sequences."""
        with pytest.raises(ValueError, match="Sequence length 1630 above maximum sequence length of 1024"):
            embeddings = embed.gen_embedding(self.df['Sequence'].tolist(), device='cuda')
            
    def test_gen_embedding(self):
        """Test esm-1b embedding with a list of sequences."""
        pp = Preprocessing(self.df)
        pp.remove_long_sequenes()
        df = pp.df
        embeddings = embed.gen_embedding(df['Sequence'].tolist(), device='cuda')
        assert embeddings is not []
        assert embeddings.shape == (8, 1280)


class TestDBCreation:
    """Test the embedding function."""
    def setup_method(self, method):
        # Replace this with the code to create your DataFrame
        self.df = Parsing.parse_tsv('tests/head_10.tsv')
        pp = Preprocessing(self.df)
        pp.remove_long_sequenes()
        self.df = pp.df

        self.qdrant = QdrantClient(path="datasets/Vector_db/")  # OR write them to disk
        self.collections_info = self.qdrant.get_collections()
        self.collection_name='pytest'
        self.embeddings = embed.gen_embedding(self.df['Sequence'].tolist(), device='cuda')

    def test_create_vector_db_collection(self):
        """Test the creation of a vector database collection."""
        embed.create_vector_db_collection(self.qdrant, self.df, self.embeddings, self.collection_name)

        assert self.collection_name in str(self.collections_info)  # is DB created?
        assert self.collections_info.vectors_count == self.df.shape[0]

    def test_load_collection_from_vector_db(self):
        """Test the loading of an existing vector database collection."""
        if self.collection_name in str(self.collections_info):  # todo: implement this nicely: access the 'name' field of the object
            entries, embeddings = embed.load_collection_from_vector_db(self.collection_name)

            assert entries is not None
            assert embeddings is not None
            assert len(entries) == self.df.shape[0]
            assert embeddings == self.embeddings
