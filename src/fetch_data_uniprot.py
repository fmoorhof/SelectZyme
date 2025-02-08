"""
Classes and functionalities around data parsing from UniProt or tabular or fasta files.
"""
import io
import logging
from requests import Session
from requests.adapters import HTTPAdapter, Retry
import re
import os
import pandas as pd


def _sanitize_filename(filename: str) -> str:
    """Remove any unsafe characters and limit to alphanumerics, underscores, or dashes. Requires more malicious input testing!!"""
    return re.sub(r'[^\w\-.]', '_', filename)


def export_annotated_fasta(df: pd.DataFrame, out_file: str):
    with open(out_file, 'w') as f_out:
        for index, row in df.iterrows():
            fasta = '>', '|'.join(row.iloc[:-1].map(str)), '\n', row.loc["sequence"], '\n'
            f_out.writelines(fasta)


class UniProtFetcher:
    """
    Retrieve tsv from uniprot (in batches if size>500).
    Look here for UniProt API help: https://www.uniprot.org/help/api_queries
    """
    def __init__(self, df_coi: list[str], out_dir: str):
        self.df_coi = df_coi
        self.out_dir = out_dir
        self.session = self._init_session()
        self.re_next_link = re.compile(r'<(.+)>; rel="next"')

    def _init_session(self):
        """
        Initializes and returns a new HTTP session with retry logic.
        The session is configured to retry requests up to 5 times with a backoff factor of 0.25
        for specific HTTP status codes (500, 502, 503, 504).
        Returns:
            Session: A configured requests.Session object.
        """

        retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
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
        coi = str(self.df_coi).strip('[]').replace("'", "")
        dfs = []
        for qry in query_terms:
            raw_data = b''
            url = f"https://rest.uniprot.org/uniprotkb/search?" \
                  f"&format=tsv" \
                  f"&query=({qry}) AND (length:[{length}])" \
                  f"&fields={coi}" \
                  f"&size=500"  # UniProt pagination to fetch more than 500 entries

            for batch, total in self._get_batch(batch_url=url):
                raw_data += batch.content  # append two bytes in python

            # Decode the raw data and create a DataFrame
            df = pd.read_csv(io.StringIO(raw_data.decode('utf-8')), delimiter='\t')

            df = self._process_dataframe(df, qry)
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)
    
    def _process_dataframe(self, df: pd.DataFrame, query_term: str) -> pd.DataFrame:
            df.columns = self.df_coi
            df['reviewed'] = ~df['reviewed'].str.contains('unreviewed')  # Set as boolean values
            df['query_term'] = query_term

            return df


    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df['accession'] != 'Entry']  # remove concatenated headers that are introduced by each query term
        logging.info(f'Total amount of retrieved entries: {df.shape[0]}')
        df.drop_duplicates(subset='accession', keep='first', inplace=True)
        df.reset_index(drop=True, inplace=True)
        # Move 'sequence' column to the end (to be avoided for .fasta export)
        sequence_col = df.pop('sequence')
        df['sequence'] = sequence_col

        logging.info(f'Total amount of non redundant entries: {df.shape[0]}')
        logging.info(f"Amount of BRENDA reviewed entries: {df['xref_brenda'].notna().sum()}")
        return df

    def save_data(self, df: pd.DataFrame, out_filename: str):
        """
        Save the given DataFrame to multiple file formats and apply specific filters.
        Parameters:
        df (pd.DataFrame): The DataFrame containing the data to be saved.
        out_filename (str): The base name for the output files.
        The method performs the following steps:
        1. Sanitizes the output filename.
        2. Saves the DataFrame to a TSV file with '_annotated.tsv' suffix.
        3. Converts the TSV file to a FASTA file with '.fasta' suffix.
        4. Writes an annotated FASTA file with '_annotated.fasta' suffix.
        # 5. Filters the DataFrame to include only reviewed entries or entries with non-null 'xref_brenda'.
        # 6. Saves the filtered DataFrame to a TSV file with '_BRENDA.tsv' suffix.
        """
        
        out_filename = _sanitize_filename(out_filename)
        df.to_csv(os.path.join(self.out_dir, out_filename + "_annotated.tsv"), sep='\t', index=False)
        export_annotated_fasta(df=df, out_file=os.path.join(self.out_dir, out_filename + '_annotated.fasta'))

        # df = df[(df['reviewed'] == True) | (df['xref_brenda'].notnull())]  # | = OR
        # df.to_csv(os.path.join(self.out_dir + out_filename + '_BRENDA_only.tsv'), sep='\t', index=False)

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
