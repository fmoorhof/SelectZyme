"""
Retrieve tsv from uniprot (in batches if size>500).

Look here for UniProt API help: https://www.uniprot.org/help/api_queries
"""
import io
import logging
from requests import Session
from requests.adapters import HTTPAdapter, Retry
import re
import os
import pandas as pd


def sanitize_filename(filename: str) -> str:
    """Remove any unsafe characters and limit to alphanumerics, underscores, or dashes"""
    return re.sub(r'[^\w\-.]', '_', filename)


class UniProtFetcher:
    def __init__(self, df_coi: list[str], out_dir: str):
        self.df_coi = df_coi
        self.out_dir = out_dir
        self.session = self._init_session()
        self.re_next_link = re.compile(r'<(.+)>; rel="next"')

    def _init_session(self):
        retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
        session = Session()
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    def tsv_to_fasta_writer(self, in_file: str, out_file: str):
        with open(in_file, 'r') as f_in, open(out_file, 'w') as f_out:
            for line in f_in:
                l = line.split('\t')
                if l[0] == 'accession':  # skip header
                    continue
                fasta = '>', l[0], '\n', l[-1]  # >Entry, Sequence
                f_out.writelines(fasta)

    def write_annotated_fasta(self, df: pd.DataFrame, out_file: str):
        with open(out_file, 'w') as f_out:
            for index, row in df.iterrows():
                fasta = '>', '|'.join(row.iloc[:-1].map(str)), '\n', row.loc["sequence"], '\n'
                f_out.writelines(fasta)

    def load_custom_csv(self, file_path: str, sep: str=';') -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path, sep=sep)
            df['reviewed'] = True
            df['sequence'] = df['sequence'].str.upper()
            df['sequence'] = df['sequence'].str.replace('<BR>', '', regex=False)  # if from tool output, replace new line characters
            if df['xref_brenda'].isnull().all():
                df['xref_brenda'] = True
            if df['ec'].isnull().all():
                df['ec'] = True
            return df
        except Exception as e:
            logging.error(f"Error loading custom CSV file: {e}")
            logging.error("Please check the custom data file provided and ensure it follows the template, avoiding characters such as tabs or ';'")
            raise

    def get_next_link(self, headers):
        if "Link" in headers:
            if match := self.re_next_link.match(headers["Link"]):
                return match.group(1)

    def get_batch(self, batch_url: str):
        while batch_url:
            response = self.session.get(batch_url)
            response.raise_for_status()
            total = response.headers["x-total-results"]
            yield response, total
            batch_url = self.get_next_link(headers=response.headers)

    def query_uniprot(self, query_terms: list[str], length: int) -> pd.DataFrame:
        coi = str(self.df_coi).strip('[]').replace("'", "")
        raw_data = b''  # byte variable initialization
        for qry in query_terms:
            url = f"https://rest.uniprot.org/uniprotkb/search?" \
                  f"&format=tsv" \
                  f"&query=({qry}) AND (length:[{length}])" \
                  f"&fields={coi}" \
                  f"&size=500"  # UniProt pagination to fetch more than 500 entries

            for batch, total in self.get_batch(batch_url=url):
                raw_data += batch.content  # append two bytes in python

        df = pd.read_csv(io.StringIO(raw_data.decode('utf-8')), delimiter='\t')
        df.columns = self.df_coi
        df['reviewed'] = ~df['reviewed'].str.contains('unreviewed')  # Set boolean values
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df['accession'] != 'Entry']  # remove concatenated headers that are introduced by each query term
        logging.info(f'Total amount of retrieved entries: {df.shape[0]}')
        df.drop_duplicates(subset='accession', keep='first', inplace=True)
        df.reset_index(drop=True, inplace=True)
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
        5. Filters the DataFrame to include only reviewed entries or entries with non-null 'xref_brenda'.
        6. Saves the filtered DataFrame to a TSV file with '_BRENDA.tsv' suffix.
        """
        
        out_filename = sanitize_filename(out_filename)
        df.to_csv(os.path.join(self.out_dir, out_filename + "_annotated.tsv"), sep='\t', index=False)
        self.tsv_to_fasta_writer(in_file=os.path.join(self.out_dir, out_filename + "_annotated.tsv"), out_file=os.path.join(self.out_dir, out_filename + '.fasta'))
        self.write_annotated_fasta(df=df, out_file=os.path.join(self.out_dir, out_filename + '_annotated.fasta'))

        df = df[(df['reviewed'] == True) | (df['xref_brenda'].notnull())]  # | = OR
        df.to_csv(os.path.join(self.out_dir + out_filename + '_BRENDA.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    df_coi = ['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence']  # xref_alphafolddb == accession

    # define data to retrieve rhia
    query_terms = ["ec:2.3.1.304", "xref%3Abrenda-2.3.1.304", "IPR006862", "rhia"]  # define your query terms for UniProt here
    length = "100 TO 1001"
    out_dir = 'datasets/output/'  # describe desired output location
    out_filename = "uniprot_rhia"

    # define data to retrieve lcp
    query_terms = ["ec:1.13.11.85", "xref%3Abrenda-1.13.11.85", "ec:1.13.11.87", "xref%3Abrenda-1.13.11.87", "ec:1.13.99.B1", "xref%3Abrenda-1.13.99.B1", "IPR037473", "IPR018713", "latex clearing protein"]  # define your query terms for UniProt here
    length = "200 TO 601"
    custom_data_location = '/raid/data/fmoorhof/PhD/SideShit/LCP/custom_seqs_no_signals.csv'  # custom_seqs_full
    out_dir = 'datasets/output/'  # describe desired output location
    out_filename = "uniprot_lcp_no_signals"    

    # define data to retrieve leFOS
    query_terms = ["ec:3.2.1.64", "xref%3Abrenda-3.2.1.64", "ec:3.2.1.65", "xref%3Abrenda-3.2.1.65", "ec:3.2.1.154", "xref%3Abrenda-3.2.1.154", "ec:3.2.1.80", "xref%3Abrenda-3.2.1.80", "IPR001362", "levanase", "levan"]  # define your query terms for UniProt here
    length = "300 TO 1020"
    custom_data_location = '/raid/data/fmoorhof/PhD/SideShit/LeFOS/custom_seqs_no_signals.csv'
    out_dir = 'datasets/output/'  # describe desired output location
    out_filename = "uniprot_lefos_no_signals"

    # define data to retrieve for exonucleases (hydrolases)
    query_terms = ["IPR001616", "IPR034720", "PF01771", "IPR011335"]
    length = "200 TO 1020"
    custom_data_location = '/raid/data/fmoorhof/PhD/SideShit/PapE/custom_seqs.csv'
    out_dir = 'datasets/output/'  # describe desired output location
    out_filename = "uniprot_PapE"

    # minimal test
    query_terms = ["ec:3.2.1.64", "PF01771"]
    length = "200 TO 1020"
    custom_data_location = '/raid/data/fmoorhof/PhD/SideShit/PapE/custom_seqs.csv'
    out_dir = 'datasets/output/'  # describe desired output location
    out_filename = "test_data"

    fetcher = UniProtFetcher(df_coi, out_dir)

    # Retrieve and save data for lcp
    df = fetcher.query_uniprot(query_terms, length)
    if custom_data_location != '':
        custom_data = fetcher.load_custom_csv(custom_data_location)
        df = pd.concat([custom_data, df], ignore_index=True)
    df = fetcher.clean_data(df)
    fetcher.save_data(df, out_filename)
