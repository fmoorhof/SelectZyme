from parsing import Parsing


def test_parse_fasta():
    """Test the parse_fasta method directly."""
    parser = Parsing('tests/head_10.fasta')
    df = parser.parse_fasta()

    assert not df.empty
    assert 'accession' in df.columns
    assert 'sequence' in df.columns

def test_parse_tsv():
    """Test the parse_tsv method directly."""
    parser = Parsing('tests/head_10.tsv')
    df = parser.parse_tsv()

    assert not df.empty
    assert 'accession' in df.columns
    assert 'sequence' in df.columns
    
def test_parse_invalid_format(invalid_file='tests/head_10.txt'):
    """Test parsing of an unsupported file format."""
    try:
        Parsing(invalid_file).parse()
    except ValueError as e:
        assert str(e) == "File format not supported."
    else:
        assert False, "ValueError not raised for unsupported file format"
