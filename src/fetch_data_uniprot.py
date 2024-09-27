"""
Retrieve tsv from uniprot (in batches if size>500).

Look here for UniProt API help: https://www.uniprot.org/help/api_queries

todo?: retrieve embeddings from uniprot, too?
"""
import requests
from requests.adapters import HTTPAdapter, Retry
import re
import io
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


def get_next_link(headers):
    if "Link" in headers:
        if match := re_next_link.match(headers["Link"]):
            return match.group(1)


def get_batch(batch_url):
    while batch_url:
        response = session.get(batch_url)
        response.raise_for_status()
        total = response.headers["x-total-results"]
        yield response, total
        batch_url = get_next_link(response.headers)


def load_custom_csv(file_path: str, df_coi: list[str]) -> pd.DataFrame:
    """load custom csv file and add it to the dataframe. The custom data need to be in the same format (column names) as the internal data.
    :param file_path: path to the custom csv file
    :param df_coi: columns of interest
    :return: df: dataframe with custom data added"""
    df = pd.read_csv(file_path, sep=',', header=None, names=df_coi, skiprows=1)
    df['reviewed'] = True
    if df['xref_brenda'].isnull().all():
        df['xref_brenda'] = True
    if df['ec'].isnull().all():
        df['ec'] = True
    return df



if __name__ == '__main__':
    # define data to retrieve
    query_terms = ["ec:2.3.1.304", "xref%3Abrenda-2.3.1.304", "IPR006862", "rhia"]  # define your query terms for UniProt here
    length = "100 TO 1001"
    out_dir = 'datasets/output/'  # describe desired output location
    out_filename = "uniprot_rhia"

    # UniProt pagination to fetch more than 500 entries
    re_next_link = re.compile(r'<(.+)>; rel="next"')
    retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))

    # Specify the columns of interest to be extracted from UniProt
    df_coi = ['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence']  # xref_alphafolddb == accession
    coi = str(df_coi).strip('[]').replace("'", "")

    raw_data = b''  # byte variable initialization
    for qry in query_terms:
        url = f"https://rest.uniprot.org/uniprotkb/search?" \
            f"&format=tsv" \
            f"&query=({qry}) AND (length:[{length}])" \
            f"&fields={coi}" \
            f"&size=500"
        for batch, total in get_batch(url):
            raw_data = raw_data + batch.content  # append two bytes in python

    df = pd.read_csv(io.StringIO(raw_data.decode('utf-8')), delimiter='\t')
    df.columns = df_coi  # UniProt sets column names differently, its easier to stick to keywords
    df['reviewed'] = ~df['reviewed'].str.contains('unreviewed')  # Set boolean values

    # Load custom data
    custom_data_location = '/raid/data/fmoorhof/PhD/SideShit/RhlA_mining/The_eight.csv'
    custom_data = load_custom_csv(custom_data_location, df_coi)
    df = pd.concat([custom_data, df], ignore_index=True)

    # cleaning data
    df = df[df['accession'] != 'Entry']  # remove concatenated headers that are introduced by each query term
    print(f'Total amount of retrieved entries: {df.shape}')
    df.drop_duplicates(subset='accession', keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f'Total amount of non redundant entries: {df.shape}')

    # save output
    df.to_csv(out_dir + out_filename + '_annotated.tsv', sep='\t', index=False)
    tsv_to_fasta_writer(in_file=out_dir + out_filename + '_annotated.tsv', out_file=out_dir + out_filename + '.fasta')
    write_annotated_fasta(df=df, out_file=out_dir + out_filename + '_annotated.fasta')

    # write again only thoose where Reviewed or EC number present
    df = df[(df['reviewed'] == True ) | (df['ec'].notnull())]  # | = OR
    print(f'Amount of SWISSProt reviewed entries: {df.shape}')
    df.to_csv(out_dir + out_filename + '_SWISSProt.tsv', sep='\t', index=False)
