from __future__ import annotations

import logging

import pandas as pd
import taxoniq


def lineage_resolver(taxid: int) -> tuple[str, str, str, list[str]]:
    """
    Retrieves the lineage of a given taxonomic identifier in taxonomic identifiers. Converts the taxonomic identifiers to scientific names and returns them as a tuple.
    The lineage is always specified with the species name first and the domain name last.

    :param taxid: NCBI taxonomic identifier
    :return: Tuple containing species, domain, and list of resolved scientific names as strings
    """
    lineage = []
    try:
        t = taxoniq.Taxon(taxid)
        for taxon in t.ranked_lineage:
            lineage.append(taxon.scientific_name)
        species = lineage[0] if lineage else "Unknown"
        domain = lineage[-1] if lineage else "Unknown"
        kingdom = lineage[-2] if lineage else "Unknown"
    except Exception:
        name = "Unknown"
        lineage = [name] * 3
        species = name
        domain = name
        kingdom = name

    return species, domain, kingdom, lineage


def set_columns_of_interest(df_cols: list) -> list:
    """
    Filters out specific columns from the DataFrame and returns a list of columns to be used for hover display in plots.
    Args:
        df (pd.DataFrame): The input DataFrame from which columns are to be filtered.
    Returns:
        list: A list of column names that are not in the predefined list of columns to avoid.
    """
    columns_to_avoid_hover = [
        "sequence",
        "BRENDA URL",
        "lineage",
        "marker_size",
        "marker_symbol",
        "selected",
        "organism_id",
    ]
    # columns_of_interest= ['accession', 'reviewed', 'ec', 'length', 'xref_brenda', 'xref_pdb', 'cluster', 'species', 'domain', 'kingdom', 'selected']
    return [col for col in df_cols if col not in columns_to_avoid_hover]


def custom_plotting(df: pd.DataFrame, size: list = [6, 8, 14], shape: list = ["circle", "diamond", "cross"]) -> pd.DataFrame:
    """
    Modify the given DataFrame before plotting to make values look nicer/custom.

    Args:
        df (pd.DataFrame): The DataFrame to be modified.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    # replace empty BRENDA entries because else they will not get plottet upon dropdown selection
    if "xref_brenda" in df.columns:
        df["xref_brenda"] = df["xref_brenda"].fillna("unknown")
        values_to_replace = ["NA", "0"]
        df["xref_brenda"] = df["xref_brenda"].replace(values_to_replace, "unknown")
        df.loc[df["xref_brenda"] != "unknown", "reviewed"] = (
            True  # add BRENDA to reviewed (not only SWISSProt)
        )
        logging.info(
            f"{(df['xref_brenda'] != 'unknown').sum()} Brenda entries are found."
        )

    # Same for UniProt EC numbers
    if "ec" in df.columns:
        df["ec"] = df["ec"].fillna("unknown")
        logging.info(f"{(df['ec'] != 'unknown').sum()} UniProt EC numbers are found.")

    # define markers for the plot
    df["marker_size"] = size[0]
    df["marker_symbol"] = shape[0]
    # overwrite defaults with custom values
    if all(col in df.columns for col in {"xref_brenda", "ec", "reviewed"}):
        condition0 = (df["reviewed"] == True) | (df["reviewed"] == "true")
        if isinstance(
            df, pd.DataFrame
        ):
            condition = df["xref_brenda"] != "unknown"
            condition2 = df["ec"] != "unknown"
        else:  # assume cudf data frame
            condition = df["xref_brenda"].to_pandas() != "unknown"
            condition2 = df["ec"].to_pandas() != "unknown"
            
        df.loc[condition2, "marker_size"] = size[1]
        df.loc[condition2, "marker_symbol"] = shape[1]  # UniProt EC numbered entries
        df.loc[condition0, "marker_size"] = size[1]
        df.loc[condition0, "marker_symbol"] = (
            shape[2]  # reviewed entries (includes if custom data is set)
        )
        df.loc[condition0 & condition, "marker_size"] = (
            size[2]  # if reviewed and BRENDA entry (usually not applies to custom data)
        )

    # provide taxonomic names and lineages from taxid (organism_id)
    if "organism_id" in df.columns:
        taxa = [lineage_resolver(i) for i in df["organism_id"].values]
        df["species"] = [tax[0] for tax in taxa]
        df["domain"] = [tax[1] for tax in taxa]
        df["kingdom"] = [tax[2] for tax in taxa]
        # df['lineage'] = [tax[3] for tax in taxa]  # full lineage

    df = df.fillna("unknown")
    df["selected"] = False

    return df



if __name__ == "__main__":
    # read example data to test script
    in_file = "tests/head_10.tsv"
    df = pd.read_csv(in_file, delimiter="\t")

    taxa = [lineage_resolver(i) for i in df["Organism (ID)"].values]
    df["species"] = [tax[0] for tax in taxa]
    df["domain"] = [tax[1] for tax in taxa]
    df["kingdom"] = [tax[2] for tax in taxa]
    df["lineage"] = [tax[3] for tax in taxa]
    print(df)
