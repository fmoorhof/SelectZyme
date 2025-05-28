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
                    marker_property: list[str],
                    size: list[int] = [6, 8, 14], 
                    shape: list[str] = ["circle", "diamond", "cross"]) -> pd.DataFrame:
    """
    Modify the given DataFrame before plotting to make values look nicer/custom.
    """
    df = _assign_marker_styles(df, marker_property, size, shape)

    if "organism_id" in df.columns:
        df = _annotate_taxonomy(df)

    # fill all NaN values with "unknown" to plot also non existent values
    df = df.fillna("unknown")
    df["selected"] = False

    return df


def _clean_column(df: pd.DataFrame, column: str, replacements: list[str] = None) -> pd.Series:
    """Helper to replace nulls and unwanted values in a column."""
    df[column] = df[column].fillna("unknown")
    if replacements:
        df[column] = df[column].replace(replacements, "unknown")
    return df[column]


def _assign_marker_styles(df: pd.DataFrame, marker_property: list[str], sizes: list[int], shapes: list[str]) -> pd.DataFrame:
    """Assign marker sizes and shapes based on marker properties set in the config.yml."""
    # Set default marker size and shape if no properties are specified
    df["marker_size"] = sizes[0]
    df["marker_symbol"] = shapes[0]

    for i, col in enumerate(marker_property):
        if not col in df.columns:
            logging.warning(f"Column {col} not found in DataFrame. Skipping marker assignment for this column.")
            continue
        
        _clean_column(df, col, replacements=["NA", "0"])
        df.loc[df[col] != "unknown", ["marker_size", "marker_symbol"]] = [sizes[i], shapes[i]]
        logging.info(f"{(df[col] != 'unknown').sum()} known {col} found.")
    
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
