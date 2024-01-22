import pytest

from embed import gen_embedding


class TestEmbedding:
    """Test the embedding function."""

    @pytest.mark.parametrize("sequences", ['MSLDTIPVVDLGPLLTGD', 'MSLDTIPVVDLGPLLTGD'])
    def test_embeddings(sequences): 
        """Test esm-1b embedding"""
        embeddings = gen_embedding(sequences, device='cuda')
        
        assert embeddings is not []
