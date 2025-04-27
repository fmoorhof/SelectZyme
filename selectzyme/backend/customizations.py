from __future__ import annotations

import logging

import pandas as pd
import taxoniq


def lineage_resolver(taxid: int) -> tuple[str, str, str, list[str]]:
    """
    Retrieves the lineage of a given taxonomic identifier in taxonomic identifiers. 
    Converts the taxonomic identifiers to scientific names and returns them as a tuple.
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
    Filters out specific columns from the DataFrame and returns
    a list of columns to be used for hover display in plots.
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
    return [col for col in df_cols if col not in columns_to_avoid_hover]


def custom_plotting(df: pd.DataFrame, 
                    size: list[int] = [6, 8, 14], 
                    shape: list[str] = ["circle", "diamond", "cross"]) -> pd.DataFrame:
    """
    Modify the given DataFrame before plotting to make values look nicer/custom.
    """
    if "xref_brenda" in df.columns:
        _clean_column(df, "xref_brenda", replacements=["NA", "0"])
        df.loc[df["xref_brenda"] != "unknown", "reviewed"] = True
        logging.info(f"{(df['xref_brenda'] != 'unknown').sum()} BRENDA entries found.")

    if "ec" in df.columns:
        _clean_column(df, "ec")
        logging.info(f"{(df['ec'] != 'unknown').sum()} UniProt EC numbers found.")

    df = _assign_marker_styles(df, size, shape)

    if "organism_id" in df.columns:
        df = _annotate_taxonomy(df)

    df = df.fillna("unknown")
    df["selected"] = False

    return df


def _clean_column(df: pd.DataFrame, column: str, replacements: list[str] = None) -> pd.Series:
    """Helper to replace nulls and unwanted values in a column."""
    df[column] = df[column].fillna("unknown")
    if replacements:
        df[column] = df[column].replace(replacements, "unknown")
    return df[column]


def _assign_marker_styles(df: pd.DataFrame, sizes: list[int], shapes: list[str]) -> pd.DataFrame:
    """Assign marker sizes and shapes based on conditions."""
    df["marker_size"] = sizes[0]
    df["marker_symbol"] = shapes[0]

    if all(col in df.columns for col in ["xref_brenda", "ec", "reviewed"]):
        condition_brenda = df["xref_brenda"] != "unknown"
        condition_ec = df["ec"] != "unknown"
        condition_reviewed = df["reviewed"].isin([True, "true"])

        df.loc[condition_ec, ["marker_size", "marker_symbol"]] = [sizes[1], shapes[1]]
        df.loc[condition_reviewed, ["marker_size", "marker_symbol"]] = [sizes[1], shapes[2]]
        df.loc[condition_reviewed & condition_brenda, "marker_size"] = sizes[2]
    
    return df


def _annotate_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """Adds species, domain, and kingdom names based on organism_id."""
    taxa = [lineage_resolver(taxid) for taxid in df["organism_id"].values]
    df["species"] = [tax[0] for tax in taxa]
    df["domain"] = [tax[1] for tax in taxa]
    df["kingdom"] = [tax[2] for tax in taxa]
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
