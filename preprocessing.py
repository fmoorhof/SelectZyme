"""
This file provides basic functionalites for preprocessing sequences e.g. file parsing
"""
import logging
from itertools import groupby


class Preprocessing:
    """
    This class should assist in the preprocessing of the data.
    e.g. parsing the fasta file.
    """
    def read_fasta(filepath):
        """
        Iteratively returns the entries from a fasta file.
        
        Parameters
        ----------
        file : str
            the fasta file
        
        Yields
        ------
        output : tuple
            a single entry in the fasta file (header, sequence)
        """
        is_header = lambda x: x.startswith('>')
        compress  = lambda x: ''.join(_.strip() for _ in x)
        reader    = iter(groupby(open(filepath), is_header))
        reader    = iter(groupby(open(filepath), is_header)) if next(reader)[0] else reader
        for key, group in reader:
            if key:
                for header in group:
                    header = header[1:].strip()
            else:
                sequence = compress(group)
                if sequence != '':
                    yield header, sequence




if __name__=='__main__':

    # example dataset from the paper
    fasta_file = 'tests/head_10.fasta'

    # todo: re-write: this is super ugly -> parse as df directly
    headers = []
    sequences = []
    for h, s in Preprocessing.read_fasta(fasta_file):
        headers.append(h)
        sequences.append(s)