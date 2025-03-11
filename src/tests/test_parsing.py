from __future__ import annotations

import tempfile
import unittest

from selectzyme.parsing import Parsing


def test_parse_fasta():
    """Test the parse_fasta method directly."""
    parser = Parsing("src/tests/head_10.fasta")
    df = parser.parse_fasta()

    assert not df.empty
    assert "accession" in df.columns
    assert "sequence" in df.columns


def test_parse_tsv():
    """Test the parse_tsv method directly."""
    parser = Parsing("src/tests/head_10.tsv")
    df = parser.parse_tsv()

    assert not df.empty
    assert "accession" in df.columns
    assert "sequence" in df.columns


def test_parse_invalid_format(invalid_file="src/tests/head_10.txt"):
    """Test parsing of an unsupported file format."""
    try:
        Parsing(invalid_file).parse()
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
        parser = Parsing(filepath)
        df = parser.parse_fasta()

        self.assertEqual(df.shape[0], 3)
        self.assertEqual(df.iloc[0]["accession"], "seq1")
        self.assertEqual(df.iloc[1]["annotation_1"], "annotation")
        self.assertEqual(df.iloc[2]["annotation_2"], "annotation2")
        self.assertEqual(df.iloc[0]["sequence"], "MAEAEMMAEA")

    def test_incomplete_annotations(self):
        fasta_content = ">seq1|ann1|\nMAEAE\nMAEAE\n>seq2\nMMAEA\n"
        filepath = create_temp_fasta(fasta_content)
        parser = Parsing(filepath)
        df = parser.parse_fasta()

        self.assertEqual(df.shape[0], 2)
        self.assertEqual(df.iloc[0]["annotation_1"], "ann1")
        self.assertIsNone(df.iloc[1]["annotation_1"])

    def test_empty_fasta(self):
        filepath = create_temp_fasta("")
        parser = Parsing(filepath)
        with self.assertRaises(ValueError):
            parser.parse_fasta()

    def test_multiline_sequence(self):
        fasta_content = ">seq1\nMAEA\nEAEA\n\n>seq2|ann\nAAAA\nCCCC\n"
        filepath = create_temp_fasta(fasta_content)
        parser = Parsing(filepath)
        df = parser.parse_fasta()

        self.assertEqual(df.iloc[0]["sequence"], "MAEAEAEA")
        self.assertEqual(df.iloc[1]["sequence"], "AAAACCCC")

    def test_irregular_spacing(self):
        fasta_content = ">seq1  \n  MAEA   \n EAEA \n>seq2|ann  \nAAAA  \nCCCC  \n"
        filepath = create_temp_fasta(fasta_content)
        parser = Parsing(filepath)
        df = parser.parse_fasta()

        self.assertEqual(df.iloc[0]["sequence"], "MAEAEAEA")
        self.assertEqual(df.iloc[1]["sequence"], "AAAACCCC")


if __name__ == "__main__":
    unittest.main()
