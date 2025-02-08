from parsing import Parsing


def test_parse_fasta(fasta_file = 'tests/head_10.fasta'):
    """Test parsing of a fasta file that consists of headers and sequences."""
    df = Parsing.parse_fasta(fasta_file)

    assert df is not []

def test_parse_tsv(tsv_file = 'tests/head_10.tsv'):
    """Test parsing of a tsv file that consists of headers and sequences."""
    df = Parsing.parse_tsv(tsv_file)

    assert df is not None