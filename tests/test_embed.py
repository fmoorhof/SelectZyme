import pytest

from embed import gen_embedding


class TestEmbedding:
    """Test the embedding function."""
    sequences = ['MSLDTIPVVDLGPLLTGD', 'MSLDTIPVVDLGPLLTGD']

    @pytest.mark.parametrize("sequences", sequences)
    def test_embeddings(self, sequences): 
        """Test esm-1b embedding"""
        seqs = sequences
        embeddings = gen_embedding(seqs, device='cuda')
        
        assert embeddings is not []
        assert embeddings.shape == (18, 1280)
