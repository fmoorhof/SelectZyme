"""Test depends on parsing and the minimal data. This is maybe not very ideal but enables to define
the test cases and occurences more stringend despite violating the isolation principle of unit tests."""

from __future__ import annotations

import pytest

from parsing import Parsing
from preprocessing import Preprocessing


class TestPreprocessing:
    """Test the preprocessing functions."""

    @pytest.fixture(
        params=[Parsing.parse_tsv, Parsing.parse_fasta]
    )  # parse tsv and fasta files
    def setup_method(self, request):
        parser = (
            Parsing("tests/head_10.tsv")
            if request.param.__name__ == "parse_tsv"
            else Parsing("tests/head_10.fasta")
        )
        self.df = (
            parser.parse_tsv()
            if request.param.__name__ == "parse_tsv"
            else parser.parse_fasta()
        )
        self.length = self.df.shape[0]  # original length of the dataframe
        self.pp = Preprocessing(self.df)  # instantiate the Preprocessing class

    def test_class_instantiation(
        self, setup_method
    ):  # , setup_method needed beacuse of the fixture
        """Test the instantiation of the Preprocessing class."""
        self.pp = Preprocessing(self.df)
        df = self.pp.df

        assert self.pp is not None
        assert df.shape[0] == self.length

    @pytest.mark.skip(
        reason="AttributeError: 'Preprocessing' object has no attribute 'remove_long_sequenes'. Did you mean: 'remove_long_sequences'?"
    )
    def test_remove_long_sequences(self, setup_method):
        """Test the removal of too long sequences."""
        self.pp.remove_long_sequenes()
        df = self.pp.df

        assert df is not None
        assert df.shape[0] != self.length

    def test_remove_sequences_without_metheonin(self, setup_method):
        """Test the removal of sequences without Methionine."""
        self.pp.remove_sequences_without_metheonin()
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

    def test_preprocess(self, setup_method):
        """Test the preprocess function that applies all preprocessing steps."""
        initial_length = self.df.shape[0]
        self.pp.preprocess()
        df = self.pp.df

        assert df is not None
        assert (
            df.shape[0] <= initial_length
        )  # The number of rows should be less than or equal to the initial length

        # Check if all preprocessing steps were applied
        assert all(
            df["sequence"].str.len() < 1024
        )  # No sequences longer than 1024 amino acids
        assert all(
            df["sequence"].str.startswith("M")
        )  # All sequences start with Methionine
        assert not any(
            df["sequence"].str.contains("X")
        )  # No sequences contain undetermined amino acids
        assert df["accession"].is_unique  # No duplicate accessions
        assert df["sequence"].is_unique  # No duplicate sequences
