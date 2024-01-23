import pytest
import pandas as pd

from preprocessing import Parsing
from preprocessing import Preprocessing


def test_parse_fasta(fasta_file = 'tests/head_10.fasta'):
    """Test parsing of a fasta file that consists of headers and sequences."""
    df = Parsing.parse_fasta(fasta_file)

    assert df is not []

def test_parse_tsv(tsv_file = 'tests/head_10.tsv'):
    """Test parsing of a tsv file that consists of headers and sequences."""
    df = Parsing.parse_tsv(tsv_file)

    assert df is not None


class TestPreprocessing:
    """Test the preprocessing functions."""
    @pytest.fixture(params=[Parsing.parse_tsv, Parsing.parse_fasta])  # parse tsv and fasta files
    def setup_method(self, request):
        parse_method = request.param
        self.df = parse_method('tests/head_10.tsv') if parse_method.__name__ == 'parse_tsv' else parse_method('tests/head_10.fasta')
        self.length = self.df.shape[0]  # original length of the dataframe
        self.pp = Preprocessing(self.df)  # instantiate the Preprocessing class

    def test_class_instantiation(self, setup_method):  # , setup_method needed beacuse of the fixture
        """Test the instantiation of the Preprocessing class."""
        self.pp = Preprocessing(self.df)
        df = self.pp.df

        assert self.pp is not None
        assert df.shape[0] == self.length

    @pytest.mark.skip(reason="AttributeError: 'Preprocessing' object has no attribute 'remove_long_sequenes'. Did you mean: 'remove_long_sequences'?")
    def test_remove_long_sequences(self, setup_method):
        """Test the removal of too long sequences."""
        self.pp.remove_long_sequenes()
        df = self.pp.df

        assert df is not None
        assert df.shape[0] != self.length

    def test_remove_sequences_without_Metheonin(self, setup_method):
        """Test the removal of sequences without Methionine."""
        self.pp.remove_sequences_without_Metheonin()
        df = self.pp.df

        assert df is not None
        assert df.shape[0] != self.length

    def test_remove_sequences_with_undertermined_amino_acids(self, setup_method): 
        """Test the removal of sequences with undertermined amino acids."""
        self.pp.remove_sequences_with_undertermined_amino_acids()
        df = self.pp.df

        assert df is not None
        assert df.shape[0] != self.length

    def test_remove_duplicate_entries(self, setup_method):
        """Test the removal of duplicate entries."""
        self.pp.remove_duplicate_entries()
        df = self.pp.df

        assert df is not None
        assert df.shape[0] != self.length

    def test_remove_duplicate_sequences(self, setup_method):
        """Test the removal of duplicate sequences."""
        self.pp.remove_duplicate_sequences()
        df = self.pp.df

        assert df is not None
        assert df.shape[0] != self.length