from __future__ import annotations

import pandas as pd
import taxoniq


def lineage_resolver(taxid: int) -> tuple[str, str, list[str]]:
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
