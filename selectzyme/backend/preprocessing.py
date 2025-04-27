from __future__ import annotations

import logging

import pandas as pd


class Preprocessing:
    """This class should assist in the preprocessing of the data.
    Instead of returning the df, the self.df gets updated"""
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def preprocess(self):
        """
        This function helps to apply all preprocessing steps to the dataframe.
        """
        logging.info(f"Number of sequences before preprocessing: {self.df.shape[0]}")
        self.remove_long_sequences()
        self.remove_sequences_without_metheonin()
        self.remove_sequences_with_undertermined_amino_acids()
        self.remove_duplicate_entries()
        self.remove_duplicate_sequences()
        logging.info(f"Number of sequences after preprocessing: {self.df.shape[0]}")

        return self.df

    def remove_long_sequences(self) -> None:
        """
        This function removes too long sequences from the dataset. 
        Sequences > 1024 amino acids cause the esm embedding to fail.
        params: df: dataframe containing the sequences
        return: df: dataframe containing only sequences with a length <= 1024 amino acids
        """
        mask = self.df["sequence"].str.len() < 1024
        self.df = self.df[mask].reset_index(drop=True)
        logging.info(
            f"{(~mask).sum()} sequences were excluded because of exaggerated size (>=1024 amino acids)"
        )

    def remove_sequences_without_metheonin(self) -> None:
        """
        This function removes sequences without a Methionine at the beginning.
        params: df: dataframe containing the sequences
        return: df: dataframe containing only sequences with a Methionine at the beginning
        """
        df = self.df[self.df["sequence"].str.startswith("M")]
        df.reset_index(drop=True, inplace=True)
        logging.info(
            f"{self.df.shape[0] - df.shape[0]} sequences were excluded because of missing Methionins."
        )
        self.df = df

    def remove_sequences_with_undertermined_amino_acids(self) -> None:
        """
        This function removes sequences with undertermined amino acids 'X'.
        params: df: dataframe containing the sequences
        return: df: dataframe containing only sequences without undertermined amino acids
        """
        df = self.df[~self.df["sequence"].str.contains("X")]
        df.reset_index(drop=True, inplace=True)
        logging.info(
            f"{self.df.shape[0] - df.shape[0]} sequences were excluded because of undertermined amino acids."
        )
        self.df = df

    def remove_duplicate_entries(self) -> None:
        """
        This function removes duplicate entries from the dataframe. In fetch_data_uniprot.py, 
        the function is already used to remove duplicates resulting from the query term. 
        However, with custom data import there can still duplicates occur that will get removed with this function.
        params: df: dataframe containing the sequences
        return: df: dataframe containing only unique sequences
        """
        df = self.df.drop_duplicates(subset="accession", keep="first")
        df.reset_index(drop=True, inplace=True)
        logging.info(
            f"{self.df.shape[0] - df.shape[0]} sequences were excluded because of duplicated accessions."
        )
        self.df = df

    def remove_duplicate_sequences(self) -> None:
        """
        This function removes duplicate entries from the dataframe.
        params: df: dataframe containing the sequences
        return: df: dataframe containing only unique sequences
        """
        df = self.df.drop_duplicates(subset="sequence", keep="first")
        df.reset_index(drop=True, inplace=True)
        logging.info(
            f"{self.df.shape[0] - df.shape[0]} sequences were excluded because of duplicates."
        )
        self.df = df