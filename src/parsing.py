import pandas as pd


class Parsing():
    """This class should assist in the parsing of the data."""
    def __init__(self, filepath: str):
        self.filepath = filepath

    def parse(self) -> pd.DataFrame:
        """Parse a file and return a dataframe."""
        if self.filepath.endswith('.tsv'):
            return self.parse_tsv()
        elif self.filepath.endswith('.csv'):
            return self.parse_csv()
        elif self.filepath.endswith('.fasta'):
            return self.parse_fasta()
        else:
            raise ValueError("File format not supported.")
                
    def parse_fasta(self) -> pd.DataFrame:
        """Parse a fasta file and return a df with the header and sequences in columns."""
        headers = []
        sequences = []
        with open(self.filepath, 'r') as file:
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
        return pd.read_csv(self.filepath, sep='\t')  # on failures, try: , encoding='ISO-8859-1'
    
    def parse_csv(self) -> pd.DataFrame:
        """Parse a tsv file and return a dataframe"""
        return pd.read_csv(self.filepath, sep=',')
    