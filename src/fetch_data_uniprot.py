"""
Retrieve tsv from uniprot (in batches if size>500).

Look here for UniProt API help: https://www.uniprot.org/help/api_queries
"""
import io
import logging

import requests
from requests.adapters import HTTPAdapter, Retry
import re
import pandas as pd


def tsv_to_fasta_writer(in_file: str, out_file: str):
    """
    Read a .tsv file and write a .fasta file.

    :param in_file:
    :param out_file:
    :return:
    """
    with open(in_file, 'r') as f_in, open(out_file, 'w') as f_out:
        for line in f_in:
            l = line.split('\t')
            if l[0] == 'accession':  # skip header
                continue
            fasta = '>', l[0], '\n', l[-1]  # >Entry, Sequence
            f_out.writelines(fasta)


def write_annotated_fasta(df: object, out_file: str):
    """
    Write a fasta file that has all the annotation in its header (needed for evolocity e.g.). The field separator in the
    header is '|' as used commonly from NCBI for .fasta sequences.

    :param out_file: file to write to disk
    :return:
    """
    with open(out_file, 'w') as f_out:
        for index, row in df.iterrows():
            fasta = '>', '|'.join(row.iloc[:-1].map(str)), '\n', row.loc["sequence"], '\n'  # row.iloc[:-1] has thrown random dtype issues: fix: .map(str)
            f_out.writelines(fasta)            


def load_custom_csv(file_path: str, df_coi: list[str]) -> pd.DataFrame:
    """load custom csv file and add it to the dataframe. The custom data need to be in the same format (column names) as the internal data.
    :param file_path: path to the custom csv file
    :param df_coi: columns of interest
    :return: df: dataframe with custom data added
    
    Problems: MS excel is using ; instead ,
    also encoding might be a problem: not utf-8 encoded -> fix: encoding='ISO-8859-1"""
    df = pd.read_csv(file_path, sep=';', header=None, names=df_coi, skiprows=1, encoding='ISO-8859-1')
    # stylistic changes and data type conversions
    df['reviewed'] = True
    df['sequence'] = df['sequence'].str.upper()
    if df['xref_brenda'].isnull().all():
        df['xref_brenda'] = True
    if df['ec'].isnull().all():
        df['ec'] = True
    return df


def get_next_link(headers, re_next_link):
    if "Link" in headers:
        if match := re_next_link.match(headers["Link"]):
            return match.group(1)


def get_batch(batch_url: str, session=requests.Session(), re_next_link: re.Pattern = None):
    while batch_url:
        response = session.get(batch_url)
        response.raise_for_status()
        total = response.headers["x-total-results"]
        yield response, total
        batch_url = get_next_link(headers=response.headers, re_next_link=re_next_link)


def query_uniprot(query_terms: list[str], df_coi: list[str], length: int) -> pd.DataFrame:
    """
    Query the UniProt database with specified terms and retrieve raw data.
    Args:
        query_terms (list[str]): A list of query terms to search in the UniProt database.
        df_coi (list[str]): A list of fields (columns of interest) to be included in the output.
        length (int): The length of the protein sequences to filter the search results.
    Returns:
        df: with columns of interest
    """

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by removing redundant entries."""
    df = df[df['accession'] != 'Entry']  # remove concatenated headers that are introduced by each query term
    logging.info(f'Total amount of retrieved entries: {df.shape}')
    df.drop_duplicates(subset='accession', keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    logging.info(f'Total amount of non redundant entries: {df.shape}')
    return df


def save_data(df: pd.DataFrame, out_dir: str, out_filename: str):
    """Save the data"""
    # save output
    df.to_csv(out_dir + out_filename + '_annotated.tsv', sep='\t', index=False)
    tsv_to_fasta_writer(in_file=out_dir + out_filename + '_annotated.tsv', out_file=out_dir + out_filename + '.fasta')
    write_annotated_fasta(df=df, out_file=out_dir + out_filename + '_annotated.fasta')

    # write again only thoose where Reviewed or EC number present
    df = df[(df['reviewed'] == True ) | (df['xref_brenda'].notnull())]  # | = OR
    print(f'Amount of SWISSProt reviewed entries: {df.shape}')
    df.to_csv(out_dir + out_filename + '_SWISSProt.tsv', sep='\t', index=False)



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
    custom_data_location = '/raid/data/fmoorhof/PhD/SideShit/LCP/custom_seqs_no_signals.csv'
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
    # define data to retrieve rhia
    query_terms = ["ec:2.3.1.304", "xref%3Abrenda-2.3.1.304", "IPR006862", "rhia"]  # define your query terms for UniProt here
    length = "100 TO 1001"
    out_dir = 'datasets/output/'  # describe desired output location
    out_filename = "uniprot_rhia"

    # define data to retrieve lcp
    query_terms = ["ec:1.13.11.85", "xref%3Abrenda-1.13.11.85", "ec:1.13.11.87", "xref%3Abrenda-1.13.11.87", "ec:1.13.99.B1", "xref%3Abrenda-1.13.99.B1", "IPR037473", "IPR018713", "latex clearing protein"]  # define your query terms for UniProt here
    length = "200 TO 601"
    out_dir = 'datasets/output/'  # describe desired output location
    out_filename = "uniprot_lcp_no_signals"    
    custom_data_location = '/raid/data/fmoorhof/PhD/SideShit/LCP/custom_seqs_no_signals.csv'  # custom_seqs_full

    df = query_uniprot(query_terms, df_coi, length)

    # Load custom data
    if custom_data_location != '':
        custom_data = load_custom_csv(custom_data_location, df_coi)
        df = pd.concat([custom_data, df], ignore_index=True)

    df = clean_data(df)
    # save_data(df, out_dir, out_filename)
