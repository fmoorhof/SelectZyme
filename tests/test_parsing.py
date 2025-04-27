"""Unit tests for the parsing module in selectzyme.backend. 
Tests for the UniProtFetcher are very difficult and i only understood the first few ones!"""
from __future__ import annotations

import gzip
import os
import tempfile
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from requests import Session

from selectzyme.backend.parsing import ParseLocalFiles, UniProtFetcher, parse_data


def test_parse_fasta():
    """Test the parse_fasta method directly."""
    parser = ParseLocalFiles("tests/head_10.fasta")
    df = parser.parse_fasta()

    assert not df.empty
    assert "accession" in df.columns
    assert "sequence" in df.columns


def test_parse_tsv():
    """Test the parse_tsv method directly."""
    parser = ParseLocalFiles("tests/head_10.tsv")
    df = parser.parse_tsv()

    assert not df.empty
    assert "accession" in df.columns
    assert "sequence" in df.columns


def test_parse_invalid_format(invalid_file="src/tests/head_10.txt"):
    """Test parsing of an unsupported file format."""
    try:
        ParseLocalFiles(invalid_file).parse()
    except ValueError as e:
        assert str(e) == "File format not supported."
    else:
        assert False, "ValueError not raised for unsupported file format"


def create_temp_fasta(content: str):
    """Helper function to create a temporary FASTA file."""
    temp = tempfile.NamedTemporaryFile(delete=False, mode="w")
    temp.write(content)
    temp.close()
    return temp.name


@pytest.fixture
def mock_parse_localfiles():
    with mock.patch("selectzyme.backend.parsing.ParseLocalFiles") as mock_class:
        instance = mock_class.return_value
        instance.parse.return_value = pd.DataFrame({"accession": ["A1", "A2"]})
        yield instance

@pytest.fixture
def mock_uniprot_fetcher():
    with mock.patch("selectzyme.backend.parsing.UniProtFetcher") as mock_class:
        instance = mock_class.return_value
        instance.query_uniprot.return_value = pd.DataFrame({"accession": ["B1", "B2"]})
        yield instance


@pytest.fixture
def tmp_out_dir(tmp_path):
    """Temporary directory to simulate the output directory."""
    return tmp_path

class TestParseData:

    def test_parse_existing_file(self, monkeypatch, tmp_out_dir, mock_parse_localfiles):
        existing_file = tmp_out_dir / "existing_project.csv"
        existing_file.write_text("dummy content")

        # Monkeypatch os.path.isfile to simulate file existence
        monkeypatch.setattr(os.path, "isfile", lambda path: True)

        df = parse_data(
            query_terms=None,
            length=300,
            custom_file="",
            existing_file=str(existing_file),
            df_coi=[]
        )

        assert isinstance(df, pd.DataFrame)
        mock_parse_localfiles.parse.assert_called_once()

    def test_parse_custom_file_only(self, monkeypatch, tmp_out_dir, mock_parse_localfiles):
        monkeypatch.setattr(os.path, "isfile", lambda path: False)

        df = parse_data(
            query_terms=None,
            length=300,
            custom_file="path/to/custom.csv",
            existing_file=str(tmp_out_dir / "nonexistent.csv"),
            df_coi=[]
        )

        assert isinstance(df, pd.DataFrame)
        assert set(df["accession"]) == {"A1", "A2"}
        mock_parse_localfiles.parse.assert_called_once()

    def test_parse_query_terms_only(self, monkeypatch, tmp_out_dir, mock_uniprot_fetcher):
        monkeypatch.setattr(os.path, "isfile", lambda path: False)

        df = parse_data(
            query_terms=["kinase", "transferase"],
            length=300,
            custom_file="",
            existing_file=str(tmp_out_dir / "nonexistent.csv"),
            df_coi=["accession", "length"]
        )

        assert isinstance(df, pd.DataFrame)
        assert set(df["accession"]) == {"B1", "B2"}
        mock_uniprot_fetcher.query_uniprot.assert_called_once_with(["kinase", "transferase"], 300)

    def test_parse_custom_and_query(self, monkeypatch, tmp_out_dir, mock_parse_localfiles, mock_uniprot_fetcher):
        monkeypatch.setattr(os.path, "isfile", lambda path: False)

        df = parse_data(
            query_terms=["hydrolase"],
            length=250,
            custom_file="path/to/custom.csv",
            existing_file=str(tmp_out_dir / "nonexistent.csv"),
            df_coi=["accession"]
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert set(df["accession"]) == {"A1", "A2", "B1", "B2"}

    def test_parse_no_input(self, monkeypatch, tmp_out_dir):
        monkeypatch.setattr(os.path, "isfile", lambda path: False)

        with pytest.raises(ValueError, match="No valid 'query_terms' or 'custom_file' provided"):
            parse_data(
                query_terms=None,
                length=100,
                custom_file="",
                existing_file=str(tmp_out_dir / "nonexistent.csv"),
                df_coi=[]
            )

    def test_parse_removes_entry_rows(self, monkeypatch, tmp_out_dir, mock_parse_localfiles, mock_uniprot_fetcher):
        """Test that 'Entry' rows are correctly removed after concat."""
        monkeypatch.setattr(os.path, "isfile", lambda path: False)

        # Mock different frames
        mock_parse_localfiles.parse.return_value = pd.DataFrame({"accession": ["Entry", "A1"]})
        mock_uniprot_fetcher.query_uniprot.return_value = pd.DataFrame({"accession": ["B1", "Entry"]})

        df = parse_data(
            query_terms=["kinase"],
            length=300,
            custom_file="path/to/custom.csv",
            existing_file=str(tmp_out_dir / "nonexistent.csv"),
            df_coi=["accession"]
        )

        assert set(df["accession"]) == {"A1", "B1"}


class TestParseFasta(unittest.TestCase):
    def test_standard_fasta(self):
        fasta_content = ">seq1\nMAEAEMMAEA\n>seq2|annotation\nMAEAE\nMMAEA\n>seq3|annotation1|annotation2\nMAEAE\nMMAEA\n"
        filepath = create_temp_fasta(fasta_content)
        parser = ParseLocalFiles(filepath)
        df = parser.parse_fasta()

        self.assertEqual(df.shape[0], 3)
        self.assertEqual(df.iloc[0]["accession"], "seq1")
        self.assertEqual(df.iloc[1]["annotation_1"], "annotation")
        self.assertEqual(df.iloc[2]["annotation_2"], "annotation2")
        self.assertEqual(df.iloc[0]["sequence"], "MAEAEMMAEA")

    def test_incomplete_annotations(self):
        fasta_content = ">seq1|ann1|\nMAEAE\nMAEAE\n>seq2\nMMAEA\n"
        filepath = create_temp_fasta(fasta_content)
        parser = ParseLocalFiles(filepath)
        df = parser.parse_fasta()

        self.assertEqual(df.shape[0], 2)
        self.assertEqual(df.iloc[0]["annotation_1"], "ann1")
        self.assertIsNone(df.iloc[1]["annotation_1"])

    def test_empty_fasta(self):
        filepath = create_temp_fasta("")
        parser = ParseLocalFiles(filepath)
        with self.assertRaises(ValueError):
            parser.parse_fasta()

    def test_multiline_sequence(self):
        fasta_content = ">seq1\nMAEA\nEAEA\n\n>seq2|ann\nAAAA\nCCCC\n"
        filepath = create_temp_fasta(fasta_content)
        parser = ParseLocalFiles(filepath)
        df = parser.parse_fasta()

        self.assertEqual(df.iloc[0]["sequence"], "MAEAEAEA")
        self.assertEqual(df.iloc[1]["sequence"], "AAAACCCC")

    def test_irregular_spacing(self):
        fasta_content = ">seq1  \n  MAEA   \n EAEA \n>seq2|ann  \nAAAA  \nCCCC  \n"
        filepath = create_temp_fasta(fasta_content)
        parser = ParseLocalFiles(filepath)
        df = parser.parse_fasta()

        self.assertEqual(df.iloc[0]["sequence"], "MAEAEAEA")
        self.assertEqual(df.iloc[1]["sequence"], "AAAACCCC")


class TestUniProtFetcher(unittest.TestCase):
    @patch("selectzyme.backend.parsing.Session")
    def setUp(self, MockSession):
        self.df_coi = [
            "accession",
            "id",
            "reviewed",
            "protein_name",
            "gene_names",
            "organism_name",
        ]
        self.fetcher = UniProtFetcher(self.df_coi)
        self.mock_session = MockSession.return_value

    def test_init_session(self):
        session = self.fetcher._init_session()
        self.assertIsInstance(session, Session)
        self.assertEqual(session.adapters["https://"].max_retries.total, 5)

    @patch("selectzyme.backend.parsing.UniProtFetcher._get_batch")
    def test_query_uniprot(self, mock_get_batch):
        mock_response = MagicMock()
        mock_response.content = gzip.compress(
            b"accession\tid\treviewed\tprotein_name\tgene_names\torganism_name\nP12345\tID1\treviewed\tProtein1\tGene1\tOrganism1\n"
        )
        mock_get_batch.return_value = [(mock_response, "1")]

        query_terms = ["kinase"]
        length = 300
        result_df = self.fetcher.query_uniprot(query_terms, length)

        expected_data = {
            "accession": ["P12345"],
            "id": ["ID1"],
            "reviewed": ["reviewed"],
            "protein_name": ["Protein1"],
            "gene_names": ["Gene1"],
            "organism_name": ["Organism1"],
            "query_term": ["kinase"],
        }
        expected_df = pd.DataFrame(expected_data)

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_get_next_link(self):
        headers = {
            "Link": '<https://rest.uniprot.org/uniprotkb/search?query=kinase&size=500&format=tsv&fields=accession,id,reviewed,protein_name,gene_names,organism_name&offset=500>; rel="next"'
        }
        next_link = self.fetcher._get_next_link(headers)
        expected_link = "https://rest.uniprot.org/uniprotkb/search?query=kinase&size=500&format=tsv&fields=accession,id,reviewed,protein_name,gene_names,organism_name&offset=500"
        self.assertEqual(next_link, expected_link)

    @patch("selectzyme.backend.parsing.UniProtFetcher._get_next_link")
    def test_get_batch(self, mock_get_next_link):
        mock_response = MagicMock()
        mock_response.headers = {"x-total-results": "1"}
        mock_response.content = b"accession\tid\treviewed\tprotein_name\tgene_names\torganism_name\nP12345\tID1\treviewed\tProtein1\tGene1\tOrganism1\n"
        self.mock_session.get.return_value = mock_response
        mock_get_next_link.return_value = None

        batch_url = "https://rest.uniprot.org/uniprotkb/search?query=kinase&size=500&format=tsv&fields=accession,id,reviewed,protein_name,gene_names,organism_name"
        batches = list(self.fetcher._get_batch(batch_url))

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][1], "1")
        self.assertEqual(batches[0][0].content, mock_response.content)


if __name__ == "__main__":
    unittest.main()
