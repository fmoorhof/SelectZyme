import pytest
import pandas as pd

from preprocessing import Parsing
from preprocessing import Preprocessing


def test_parse_fasta(fasta_file = 'tests/head_10.fasta'):
    """Test parsing of a fasta file that consists of headers and sequences."""
    headers, sequences = Parsing.parse_fasta(fasta_file)

    assert headers is not []
    assert sequences is not []

def test_parse_tsv(tsv_file = 'tests/head_10.tsv'):
    """Test parsing of a tsv file that consists of headers and sequences."""
    df = Parsing.parse_tsv(tsv_file)

    assert df is not None


class TestPreprocessingFASTA:
    """Test the preprocessing functions."""
    def setup_method(self, method):
        # Replace this with the code to create your DataFrame
        headers, sequences = Parsing.parse_fasta('tests/head_10.fasta')
        self.df = pd.DataFrame({'Header': headers, 'Sequence': sequences})
        self.length = self.df.shape[0]  # original length of the dataframe
        self.preprocessing = Preprocessing(self.df)  # instantiate the Preprocessing class

    def test_remove_long_sequences(self):
        """Test the removal of too long sequences."""
        df = self.preprocessing.remove_long_sequenes()

        assert df is not None
        assert df.shape[0] != self.length

    def test_remove_sequences_without_Metheonin(self):
        """Test the removal of sequences without Methionine."""
        df = self.preprocessing.remove_sequences_without_Metheonin()

        assert df is not None
        assert df.shape[0] != self.length

    def test_remove_sequences_with_undertermined_amino_acids(self): 
        """Test the removal of sequences with undertermined amino acids."""
        df = self.preprocessing.remove_sequences_with_undertermined_amino_acids()

        assert df is not None
        assert df.shape[0] != self.length


class TestPreprocessingTSV:
    """Test the preprocessing functions."""
    def setup_method(self, method):
        # Replace this with the code to create your DataFrame
        self.df = Parsing.parse_tsv('tests/head_10.tsv')
        self.length = self.df.shape[0]  # original length of the dataframe
        self.preprocessing = Preprocessing(self.df)  # instantiate the Preprocessing class

    def test_remove_long_sequences(self):
        """Test the removal of too long sequences."""
        df = self.preprocessing.remove_long_sequenes()

        assert df is not None
        assert df.shape[0] != self.length

    def test_remove_sequences_without_Metheonin(self):
        """Test the removal of sequences without Methionine."""
        df = self.preprocessing.remove_sequences_without_Metheonin()

        assert df is not None
        assert df.shape[0] != self.length

    def test_remove_sequences_with_undertermined_amino_acids(self): 
        """Test the removal of sequences with undertermined amino acids."""
        df = self.preprocessing.remove_sequences_with_undertermined_amino_acids()

        assert df is not None
        assert df.shape[0] != self.length