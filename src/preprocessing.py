"""
This file provides basic functionalites for preprocessing sequences and file parsing from .fasta, .tsv and .csv files.
"""
import logging

import pandas as pd


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class Parsing():
    """This class should assist in the parsing of the data."""
    def parse_fasta(filepath: str) -> pd.DataFrame:
        """Parse a fasta file and return a df with the header and sequences in columns.
        params: filepath: path to the fasta file
        return: headers: list of headers
        return: sequences: list of sequences"""
        headers = []
        sequences = []
        with open(filepath, 'r') as file:
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

        df = pd.DataFrame({'accession': headers, 'sequence': sequences})
        return df
    
    def parse_tsv(filepath: str) -> pd.DataFrame:
        """Parse a tsv file and return a dataframe.
        params: filepath: path to the tsv file
        return: df: dataframe containing the sequences"""
        df = pd.read_csv(filepath, sep='\t')
        return df    
    
    def parse_csv(filepath: str) -> pd.DataFrame:
        """Parse a tsv file and return a dataframe.
        params: filepath: path to the tsv file
        return: df: dataframe containing the sequences"""
        df = pd.read_csv(filepath, sep=',')
        return df 
    

class Preprocessing:
    """This class should assist in the preprocessing of the data. Instead of returning the df, the self.df gets updated"""
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def remove_long_sequences(self) -> None:
        """
        This function removes too long sequences from the dataset. Sequences > 1024 amino acids cause the esm embedding to fail.
        params: df: dataframe containing the sequences
        return: df: dataframe containing only sequences with a length <= 1024 amino acids
        """
        mask = self.df['sequence'].str.len() < 1024
        self.df = self.df[mask].reset_index(drop=True)
        logging.info(f'{(~mask).sum()} sequences were excluded because of exaggerated size (>=1024 amino acids)')

    def remove_sequences_without_Metheonin(self) -> None:
        """
        This function removes sequences without a Methionine at the beginning.
        params: df: dataframe containing the sequences
        return: df: dataframe containing only sequences with a Methionine at the beginning
        """
        df = self.df[self.df['sequence'].str.startswith('M')]
        df.reset_index(drop=True, inplace=True)
        logging.info(f'{self.df.shape[0]-df.shape[0]} sequences were excluded because of missing Methionins.')
        self.df = df
    
    def remove_sequences_with_undertermined_amino_acids(self) -> None:
        """
        This function removes sequences with undertermined amino acids 'X'.
        params: df: dataframe containing the sequences
        return: df: dataframe containing only sequences without undertermined amino acids
        """
        df = self.df[~self.df['sequence'].str.contains('X')]
        df.reset_index(drop=True, inplace=True)
        logging.info(f'{self.df.shape[0]-df.shape[0]} sequences were excluded because of undertermined amino acids.')
        self.df = df

    def remove_duplicate_entries(self) -> None:
        """
        This function removes duplicate entries from the dataframe. In fetch_data_uniprot.py, the function is already used to remove duplicates resulting from the query term. However, with custom data import there can still duplicates occur that will get removed with this function.
        params: df: dataframe containing the sequences
        return: df: dataframe containing only unique sequences
        """
        df = self.df.drop_duplicates(subset='accession', keep='first')
        df.reset_index(drop=True, inplace=True)
        logging.info(f'{self.df.shape[0]-df.shape[0]} sequences were excluded because of duplicated accessions.')
        self.df = df

    def remove_duplicate_sequences(self) -> None:
        """
        This function removes duplicate entries from the dataframe.
        params: df: dataframe containing the sequences
        return: df: dataframe containing only unique sequences
        """
        df = self.df.drop_duplicates(subset='sequence', keep='first')
        df.reset_index(drop=True, inplace=True)
        logging.info(f'{self.df.shape[0]-df.shape[0]} sequences were excluded because of duplicates.')
        self.df = df      



if __name__=='__main__':

    # example datasets
    fasta_file = 'tests/head_10.fasta'
    # fasta_file = 'tests/head_10.tsv'

    if fasta_file.endswith('.fasta'):
        df = Parsing.parse_fasta(fasta_file)
    else:
        df = Parsing.parse_tsv(fasta_file)
    
    pp = Preprocessing(df)
    pp.remove_long_sequences()
    pp.remove_sequences_without_Metheonin()
    pp.remove_sequences_with_undertermined_amino_acids()
    pp.remove_duplicate_entries()
    pp.remove_duplicate_sequences()
    print(df.shape)
    df = pp.df  # Get the final DataFrame
    print(df.shape)
