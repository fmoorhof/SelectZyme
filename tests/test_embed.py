import pytest

from embed import read_fasta
from embed import gen_embedding


def test_reading():
    # minimal example data set
    fasta_file = 'head_10.fasta'

    # todo: re-write: this is super ugly -> parse as df directly
    # get the headers and sequences from the fasta file
    headers = []
    sequences = []
    for h, s in read_fasta(fasta_file):
        headers.append(h)
        sequences.append(s)

    assert headers is not []
    assert sequences is not []


def test_embeddings(sequences): 
    embeddings = gen_embedding(sequences, device='cuda')
    
    assert embeddings is not []
