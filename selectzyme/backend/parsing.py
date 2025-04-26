from __future__ import annotations

import logging
import os
from collections import defaultdict
from gzip import decompress
from io import StringIO
from re import compile
from urllib.parse import quote_plus

import pandas as pd
from requests import Session
from requests.adapters import HTTPAdapter, Retry


def parse_data(project_name, query_terms, length, custom_file, out_dir, df_coi):
    existing_file = os.path.join(out_dir, project_name + ".csv")
    df = _parse_data(existing_file, custom_file, query_terms, length, df_coi)
    df = _clean_data(df)
    return df


def _parse_data(
    existing_file: str, custom_file: str, query_terms: list, length: str, df_coi: list
):
    if os.path.isfile(existing_file):
        return ParseLocalFiles(existing_file).parse()

    if (
        query_terms != ""
    ):  # todo: handle if query_terms NoneType -> breaks execution when query_terms not defined in config.yml
        fetcher = UniProtFetcher(df_coi)
    if custom_file != "":
        df_custom = ParseLocalFiles(custom_file).parse()
        if query_terms == "":
            return df_custom
        df = fetcher.query_uniprot(query_terms, length)
        df = pd.concat(
            [df_custom, df], ignore_index=True
        )  # custom data first that they are displayed first in plot legends
        return df
    elif query_terms != "":
        return fetcher.query_uniprot(query_terms, length)
    else:
        raise ValueError("Either 'query_terms' or 'custom_file' must be provided.")

 
def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        df["accession"] != "Entry"
    ]  # remove concatenated headers that are introduced by each query term
    logging.info(f"Total amount of retrieved entries: {df.shape[0]}")
    df.drop_duplicates(subset="accession", keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logging.info(f"Total amount of non redundant entries: {df.shape[0]}")
    if "xref_brenda" in df.columns:
        logging.info(
            f"Amount of BRENDA reviewed entries: {df['xref_brenda'].notna().sum()}"
        )
    return df


class ParseLocalFiles:
    "This class parses TSV, CSV, and FASTA files into pandas DataFrames."
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
        try:
            return pd.read_csv(self.filepath, sep="\t")
        except UnicodeDecodeError:
            return pd.read_csv(self.filepath, sep="\t", encoding="ISO-8859-1")

    def parse_csv(self) -> pd.DataFrame:
        """Parse a csv file and return a dataframe
        Automatically detects separator (comma or semicolon) and handles common encoding issues.
        """
        # First try UTF-8 encoding with both separators
        for sep in [",", ";"]:
            try:
                df = pd.read_csv(self.filepath, sep=sep, encoding='utf-8')
                # Verify we got more than one column (simple separator detection)
                if len(df.columns) > 1:
                    return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                pass
        
        # If UTF-8 failed, try ISO-8859-1 encoding with both separators
        for sep in [",", ";"]:
            try:
                df = pd.read_csv(self.filepath, sep=sep, encoding='ISO-8859-1')
                if len(df.columns) > 1:
                    return df
            except pd.errors.ParserError:
                pass
        
        # If all attempts failed, try with no separator specified (pandas auto-detection)
        try:
            return pd.read_csv(self.filepath, encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(self.filepath, encoding='ISO-8859-1')
        
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
                    sequences[current_header] += line  # Append sequence to the corresponding header

        if not headers:
            raise ValueError("FASTA file is empty or incorrectly formatted.")

        # Parse headers and extract annotations
        parsed_headers = [header.split("|") for header in headers]
        max_annotations = max(len(h) for h in parsed_headers)

        # Create DataFrame with numbered annotation columns
        columns = ["accession"] + [f"annotation_{i + 1}" for i in range(max_annotations - 1)] + ["sequence"]
        data = [h + [None] * (max_annotations - len(h)) + [sequences["|".join(h)]] for h in parsed_headers]

        return pd.DataFrame(data, columns=columns)


class UniProtFetcher:
    """
    Retrieve tsv from uniprot (in batches if size>500).
    Look here for UniProt API help: https://www.uniprot.org/help/api_queries
    """

    def __init__(self, df_coi: list[str]):
        self.df_coi = df_coi
        self.session = self._init_session()
        self.re_next_link = compile(r'<(.+)>; rel="next"')

    def _init_session(self):
        """
        Initializes and returns a new HTTP session with retry logic.
        The session is configured to retry requests up to 5 times with a backoff factor of 0.25
        for specific HTTP status codes (500, 502, 503, 504).
        Returns:
            Session: A configured requests.Session object.
        """

        retries = Retry(
            total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504]
        )
        session = Session()
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    def query_uniprot(self, query_terms: list[str], length: int) -> pd.DataFrame:
        """
        Queries the UniProt database with the given query terms and sequence length, and returns the results as a pandas DataFrame.
        Args:
            query_terms (list[str]): A list of query terms to search for in the UniProt database.
            length (int): The length of the protein sequences to filter by.
        Returns:
            pd.DataFrame: A DataFrame containing the query results with columns specified in self.df_coi. The 'reviewed' column is a boolean indicating whether the entry is reviewed.
        Raises:
            ValueError: If the query to UniProt fails or returns an error.
        """
        coi = str(self.df_coi).strip("[]").replace("'", "")
        dfs = []
        for qry in query_terms:
            raw_data = b""
            # Encode query and fields parameters to avoid special character issues
            encoded_query = quote_plus(f"{qry} AND length:[{length}]")
            encoded_fields = quote_plus(coi)
            url = (
                f"https://rest.uniprot.org/uniprotkb/search?"
                f"&format=tsv"
                f"&query={encoded_query}"
                f"&fields={encoded_fields}"
                f"&compressed=true"
                f"&size=500"
            )  # UniProt pagination to fetch more than 500 entries

            for batch, total in self._get_batch(batch_url=url):
                if int(total) > 100000:
                    logging.warning(
                        f"Query term '{qry}' skipped: Exceeds maximum allowed entries (100,000) per query term. Total entries: {total}. You might want to specify the query term more specifically."
                    )
                    continue
                raw_data += batch.content

            logging.info(f"Retrieved {total} entries for query term: {qry}")

            decompressed_data = decompress(raw_data).decode(
                "utf-8"
            )  # Decompress raw data
            df = pd.read_csv(StringIO(decompressed_data), delimiter="\t")

            df = self._process_dataframe(df, qry)
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def _process_dataframe(self, df: pd.DataFrame, query_term: str) -> pd.DataFrame:
        df.columns = self.df_coi
        df["reviewed"] = ~df["reviewed"].str.contains(
            "unreviewed"
        )  # Set as boolean values
        df["query_term"] = query_term

        return df

    def _get_next_link(self, headers):
        if "Link" in headers:
            if match := self.re_next_link.match(headers["Link"]):
                return match.group(1)

    def _get_batch(self, batch_url: str):
        while batch_url:
            response = self.session.get(batch_url)
            response.raise_for_status()
            total = response.headers["x-total-results"]
            yield response, total
            batch_url = self._get_next_link(headers=response.headers)
