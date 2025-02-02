import logging

import pandas as pd
import cudf

from src.ncbi_taxonomy_resolver import lineage_resolver


def set_columns_of_interest(df_cols: list) -> list:
    """
    Filters out specific columns from the DataFrame and returns a list of columns to be used for hover display in plots.
    Args:
        df (pd.DataFrame): The input DataFrame from which columns are to be filtered.
    Returns:
        list: A list of column names that are not in the predefined list of columns to avoid.
    """
    columns_to_avoid_hover = ['sequence', 'BRENDA URL', 'lineage', 'marker_size', 'marker_symbol', 'selected', 'organism_id']
    # columns_of_interest= ['accession', 'reviewed', 'ec', 'length', 'xref_brenda', 'xref_pdb', 'cluster', 'species', 'domain', 'kingdom', 'selected']
    return [col for col in df_cols if col not in columns_to_avoid_hover]


def custom_plotting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify the given DataFrame before plotting to make values look nicer/custom.

    Args:
        df (pd.DataFrame): The DataFrame to be modified.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    # replace empty ECs because they will not get plottet (if color='ec' or 'xref_brenda')
    df['xref_brenda'] = df['xref_brenda'].fillna('')
    values_to_replace = ['NA', '0']
    df['xref_brenda'] = df['xref_brenda'].replace(values_to_replace, '')
    
    df['ec'] = df['ec'].fillna('unknown')  # replace empty ECs because they will not get plottet (if color='ec')
    df['ec'] = df['ec'].str.replace(r'\..\..\..\.-;', '', regex=True)  # 1.1.1.- to 0.0.0.0    
    df['ec'] = df['ec'].str.replace(r'.*\..*\..*\.-; ?|; .*\..*\..*\.-', '', regex=True)  # extract only complete ec of 1.14.11.-; -.1.11.-; 1.14.11.29; X.-.11.-
    logging.info(f"{(df['ec'] != 'unknown').sum()} UniProt EC numbers are found.")
    logging.info(f"{(df['xref_brenda'] != '').sum()} Brenda entries are found.")

    # define markers for the plot
    if isinstance(df, cudf.DataFrame):  # fix for AttributeError: 'Series' object has no attribute 'to_pandas' (cudf vs. pandas)
        condition = (df['reviewed'] == True) | (df['reviewed'] == 'true')
        condition2 = (df['ec'].to_pandas() != 'unknown')
    else:  # pandas DataFrame
        condition = (df['reviewed'] == True) | (df['reviewed'] == 'true')
        condition2 = (df['ec'] != 'unknown')
    df['marker_size'] = 6
    df['marker_symbol'] = 'circle'
    df.loc[condition2, 'marker_size'] = 8 # Set to other value for data points that meet the condition
    df.loc[condition2, 'marker_symbol'] = 'diamond'
    df.loc[condition, 'marker_size'] = 10
    df.loc[condition, 'marker_symbol'] = 'cross'
    # df.loc[condition & condition2, 'marker_size'] = 14  # 2 conditions possible

    # provide taxonomic names and lineages from taxid (organism_id)
    taxa = [lineage_resolver(i) for i in df['organism_id'].values]
    df['species'] = [tax[0] for tax in taxa]
    df['domain'] = [tax[1] for tax in taxa]
    df['kingdom'] = [tax[2] for tax in taxa]
    # df['lineage'] = [tax[3] for tax in taxa]  # full lineage

    df['selected'] = False
    df.loc[df['xref_brenda'] != '', 'reviewed'] = True  # add BRENDA to reviewed (not only SWISSProt)

    # todo: remove later
    # df['activity_on_PET'] = df['activity_on_PET'].apply(lambda x: True if x == 1.0 else False)
    return df