
from __future__ import annotations

import gzip
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from requests import Session

from selectzyme.backend.parsing import ParseLocalFiles, UniProtFetcher


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


# todo: logic un-tested yet. So far i see no real call to uniprot is done. change this and assert delivered response.
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
            "reviewed": [True],
            "protein_name": ["Protein1"],
            "gene_names": ["Gene1"],
            "organism_name": ["Organism1"],
            "query_term": ["kinase"],
        }
        expected_df = pd.DataFrame(expected_data)

        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_process_dataframe(self):
        data = {
            "accession": ["P12345"],
            "id": ["ID1"],
            "reviewed": ["reviewed"],
            "protein_name": ["Protein1"],
            "gene_names": ["Gene1"],
            "organism_name": ["Organism1"],
        }
        df = pd.DataFrame(data)
        query_term = "kinase"
        processed_df = self.fetcher._process_dataframe(df, query_term)

        expected_data = {
            "accession": ["P12345"],
            "id": ["ID1"],
            "reviewed": [True],
            "protein_name": ["Protein1"],
            "gene_names": ["Gene1"],
            "organism_name": ["Organism1"],
            "query_term": ["kinase"],
        }
        expected_df = pd.DataFrame(expected_data)

        pd.testing.assert_frame_equal(processed_df, expected_df)

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
