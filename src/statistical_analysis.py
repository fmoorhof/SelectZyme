"""Perform a statistical analysis on the data. Explanatory data analysis (EDA)"""
import logging

import taxoniq
import plotly.graph_objects as go
import pandas as pd


logging.basicConfig(format='%(levelname)-8s| %(module)s.%(funcName)s: %(message)s', level=logging.DEBUG)
global exc
exc = []


def statistical_analysis(df):
    """Perform counting of EC numbers"""
    value_counts = df['BRENDA'].value_counts().reset_index()
    value_counts.columns = ['BRENDA', 'Count']
 
    value_counts.to_csv('Output/brenda_ec_count.csv', index=False)


def scientific_name_resolver(taxid: int) -> str:
    """
    returns scientific names for given NCBI taxonomic identifiers. Many IDs can not be resolved hence the try except statement

    :param taxid: NCBI taxonomic identifier
    :return: string of scientific names
    """
    try:
        t = taxoniq.Taxon(taxid)
        name = t.scientific_name
    except:
        name = 'cellular organisms'
        logging.info(taxid, 'could not be resolved and was set to: ', name)
        exc.append(taxid)

    return name


def lineage_resolver(taxid: int, level: int) -> list:
    """
    Returns scientific names for given NCBI taxonomic identifiers.

    :param taxid: NCBI taxonomic identifier
    :param level: Number of levels to include in the resolved scientific names
    :return: List of resolved scientific names
    """
    res = []
    try:
        t = taxoniq.Taxon(taxid)
        for i in t.ranked_lineage:
            res.append(i.scientific_name)
        res = res[-level:]
    except:
        name = 'cellular organisms'
        res = [name]*3

    return res


def inintialize_sankey_data(nodes: list) -> dict:
    """
    Initialize the data for the Sankey plot.

    Args:
        nodes (list): A list of nodes for the Sankey plot.

    Returns:
        dict: A dictionary containing the initialized data for the Sankey plot.
    """
    sankey_data = {
        'type': 'sankey',
        'orientation': 'h',
        'node': {
            'pad': 15,
            'thickness': 20,
            'line': {
                'color': 'black',
                'width': 0.5
            },
            'label': nodes
        },
        'link': {
            'source': [],
            'target': [],
            'value': []
        }
    }
    return sankey_data


def create_nodes_links(df, sankey_data: dict) -> dict:
    """
    Create nodes and links for the Sankey plot.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the lineage information.
        sankey_data (dict): The dictionary to store the Sankey plot data.

    Returns:
        dict: The updated sankey_data dictionary with nodes and links.

    """
    node_dict = {}  # store node names and their corresponding indices
    for lineage in df['Lineage']:
        source = None
        for taxon in lineage:
            if taxon not in node_dict:
                node_dict[taxon] = len(node_dict)
                nodes.append(taxon)

            target = node_dict[taxon]

            if source is not None:
                sankey_data['link']['source'].append(source)
                sankey_data['link']['target'].append(target)
                sankey_data['link']['value'].append(1)

            source = target

    return sankey_data



if __name__ == "__main__":
    # read example data to test script
    in_file = 'tests/head_10.tsv'
    in_file = '/raid/data/fmoorhof/PhD/Data/SKD001_Literature_Mining/Batch5/batch5_annotated.tsv'
    df = pd.read_csv(in_file, delimiter='\t', usecols=['Organism (ID)'])
    df = df.head(300)
    
    # df_counts = pd.DataFrame(df['Organism (ID)'].groupby(df['Organism (ID)'].tolist()).size().reset_index(name='size'))  # count and make unique
    df = df.groupby(df.columns.tolist(), as_index=False).size()  # count and make unique
    df['Organism Name'] = [scientific_name_resolver(i) for i in df['Organism (ID)'].values]
    df['Lineage'] = [lineage_resolver(i, level=0) for i in df['Organism (ID)'].values]
    logging.info(f"The following {len(exc)}/{len(df['Organism (ID)'])} entries were discovered to be unresolvable: {exc}."
                f"You can try to resolve them manually with the NCBI taxonomy browser")

    # todo: use different colors, make axis title, consider amount of data to display
    x_vals = ['Life', 'Domain', 'Kingdom', 'Phylum', 'Species']  # todo: no axis title yet implemented
    nodes = []
    sankey_data = inintialize_sankey_data(nodes)
    sankey_data = create_nodes_links(df, sankey_data)

    # todo: find way to filter ugly data that are not named correctly
    fig = go.Figure(data=[sankey_data])
    fig.update_layout(title_text="NCBI Taxonomy Lineage Sankey Plot")
    fig.show(host='0.0.0.0', port=8051)
