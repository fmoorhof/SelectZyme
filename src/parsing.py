import pandas as pd


class Parsing():
    """This class should assist in the parsing of the data."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        
    def parse_fasta(self) -> pd.DataFrame:
        """Parse a fasta file and return a df with the header and sequences in columns."""
        headers = []
        sequences = []
        with open(self, 'r') as file:
            sequence = ""
            for line in file:
                if line.startswith('>'):
                    if sequence != "":
                        sequences.append(sequence)
                        sequence = ""
                    headers.append(line.strip())
                else:
                    sequence += line.strip()
            sequences.append(sequence)  # Append the last sequence

        return pd.DataFrame({'accession': headers, 'sequence': sequences})
    
    def parse_tsv(self) -> pd.DataFrame:
        """Parse a tsv file and return a dataframe."""
        return pd.read_csv(self, sep='\t')
    
    def parse_csv(self) -> pd.DataFrame:
        """Parse a tsv file and return a dataframe"""
        return pd.read_csv(self, sep=',')
    