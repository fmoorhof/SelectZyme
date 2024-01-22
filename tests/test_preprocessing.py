import pytest

from preprocessing import read_fasta


class TestPreprocessing:
    """Test the preprocessing function."""
    def test_reading(fasta_file = 'head_10.fasta'):
        """Test reading of a fasta file that consists of headers and sequences."""
        headers = []
        sequences = []
        for h, s in read_fasta(fasta_file):
            headers.append(h)
            sequences.append(s)

        assert headers is not []
        assert sequences is not []
