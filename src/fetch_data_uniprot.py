"""
Classes and functionalities around data parsing from UniProt or tabular or fasta files.
"""
import io
from requests import Session
from requests.adapters import HTTPAdapter, Retry
import re
import pandas as pd


class UniProtFetcher:
    """
    Retrieve tsv from uniprot (in batches if size>500).
    Look here for UniProt API help: https://www.uniprot.org/help/api_queries
    """
    def __init__(self, df_coi: list[str]):
        self.df_coi = df_coi
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
