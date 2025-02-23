import logging
from collections import defaultdict

import pandas as pd


class Parsing:
    """This class should assist in the parsing of the data."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def parse(self) -> pd.DataFrame:
        """Parse a file and return a dataframe."""
        if self.filepath.endswith(".tsv"):
            return self.parse_tsv()
        elif self.filepath.endswith(".csv"):
            return self.parse_csv()
        elif self.filepath.endswith(".fasta"):
            return self.parse_fasta()
        else:
            raise ValueError("File format not supported.")

    def parse_tsv(self) -> pd.DataFrame:
        """Parse a tsv file and return a dataframe."""
        return pd.read_csv(
            self.filepath, sep="\t"
        )  # on failures, try: , encoding='ISO-8859-1'

    def parse_csv(self) -> pd.DataFrame:
        """Parse a tsv file and return a dataframe"""
        return pd.read_csv(self.filepath, sep=",")

    def parse_fasta(self) -> pd.DataFrame:
        """
        Parse a FASTA file and return a DataFrame with headers, annotations, and sequences.

        Parameters:
        -----------
        filepath : str
            Path to the FASTA file.

        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: 'accession', 'sequence', and numbered annotation columns.
        """
        headers = []
        sequences = defaultdict(str)

        with open(self.filepath, "r") as file:
            current_header = None
            for line in file:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                if line.startswith(">"):
                    current_header = line.lstrip(">")
                    headers.append(current_header)
                elif current_header:
                    sequences[current_header] += (
                        line  # Append sequence to the corresponding header
                    )

        if not headers:
            raise ValueError("FASTA file is empty or incorrectly formatted.")

        # Parse headers and extract annotations
        parsed_headers = [header.split("|") for header in headers]
        max_annotations = max(len(h) for h in parsed_headers)

        # Create DataFrame with numbered annotation columns
        columns = (
            ["accession"]
            + [f"annotation_{i + 1}" for i in range(max_annotations - 1)]
            + ["sequence"]
        )
        data = [
            h + [None] * (max_annotations - len(h)) + [sequences["|".join(h)]]
            for h in parsed_headers
        ]

        return pd.DataFrame(data, columns=columns)
