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



if __name__ == '__main__':
    # define data to retrieve
    ec_list = ["ec:1.13.11.85", "xref%3Abrenda-1.13.11.85", "ec:1.13.11.87", "xref%3Abrenda-1.13.11.87"]  # ec number and brenda retrieval
    ipr_list = ["IPR037473", "IPR018713", "latex clearing protein"]
    length = "200 TO 601"
    out_dir = 'datasets/output/'  # save output here

    # UniProt pagination to fetch more than 500 entries
    re_next_link = re.compile(r'<(.+)>; rel="next"')
    retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))

    # Specify the columns of interest to be extracted from UniProt
    df_coi = ['accession', 'reviewed', 'ec', 'organism_id', 'length', 'xref_brenda', 'xref_pdb', 'sequence']  # xref_alphafolddb == accession
    coi = str(df_coi).strip('[]').replace("'", "")

    raw_data = b''  # byte variable initialization
    for ipr in ec_list+ipr_list:
        url = f"https://rest.uniprot.org/uniprotkb/search?" \
            f"&format=tsv" \
            f"&query=({ipr}) AND (length:[{length}])" \
            f"&fields={coi}" \
            f"&size=500"
        for batch, total in get_batch(url):
            raw_data = raw_data + batch.content  # append two bytes in python

    df = pd.read_csv(io.StringIO(raw_data.decode('utf-8')), delimiter='\t')
    df.columns = df_coi  # UniProt sets column names differently, its easier to stick to keywords
    print(df.shape)
    df = df[df['accession'] != 'Entry']  # remove concatenated headers that are introduced by each query term
    print(df.shape)
    df.drop_duplicates(subset='accession', keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df.shape)

    # Set bool
    df['reviewed'] = ~df['reviewed'].str.contains('unreviewed')

    # save output
    df.to_csv(out_dir + 'uniprot_lcp_annotated.tsv', sep='\t', index=False)
    tsv_to_fasta_writer(in_file=out_dir + 'uniprot_lcp_annotated.tsv', out_file=out_dir + 'uniprot_lcp.fasta')
    write_annotated_fasta(df=df, out_file=out_dir + 'uniprot_lcp_annotated.fasta')

    # write again only thoose where Reviewed or EC number present
    df = df[(df['reviewed'] == True ) | (df['ec'].notnull())]  # | = OR
    print(df.shape)
    df.to_csv(out_dir + 'uniprot_lcp_SWISSProt.tsv', sep='\t', index=False)
