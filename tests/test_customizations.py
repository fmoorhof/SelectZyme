from __future__ import annotations

import pandas as pd

from selectzyme.backend.customizations import (
    custom_plotting,
    lineage_resolver,
    set_columns_of_interest,
)


def test_lineage_resolver_valid_taxid():
    species, domain, kingdom, lineage = lineage_resolver(
        9606
    )  # Example taxid for Homo sapiens
    assert species == "Homo sapiens"
    assert domain == "Eukaryota"
    assert kingdom == "Metazoa"
    assert "Homo sapiens" in lineage


def test_lineage_resolver_invalid_taxid():
    species, domain, kingdom, lineage = lineage_resolver(-1)  # Invalid taxid
    assert species == "Unknown"
    assert domain == "Unknown"
    assert kingdom == "Unknown"
    assert lineage == ["Unknown", "Unknown", "Unknown"]


def test_lineage_resolver_empty_taxid():
    species, domain, kingdom, lineage = lineage_resolver(0)  # Edge case for empty taxid
    assert species == "Unknown"
    assert domain == "Unknown"
    assert kingdom == "Unknown"
    assert lineage == ["Unknown", "Unknown", "Unknown"]


def test_set_columns_of_interest():
    df_cols = ["accession", "sequence", "organism", "marker_size"]
    filtered_cols = set_columns_of_interest(df_cols)
    assert "sequence" not in filtered_cols
    assert "marker_size" not in filtered_cols
    assert "accession" in filtered_cols
    assert "organism" in filtered_cols


def test_custom_plotting_assigns_markers_and_clean_values():
    # Minimal DataFrame for testing
    df = pd.DataFrame({
        "xref_brenda": ["P12345", None, "NA", "0", "[]"],
        "ec": ["1.1.1.1", None, "2.2.2.2", None, "[]"],
        "reviewed": [True, False, "true", False, "[]"],
        "organism_id": [9606, 9606, -1, 0, "[]"],
    })

    processed_df = custom_plotting(df, marker_property=["xref_brenda", "ec"])

    # Test fillna replacements
    assert all(processed_df["xref_brenda"].isin(["P12345", "unknown"]))
    assert all(processed_df["ec"].isin(["1.1.1.1", "2.2.2.2", "unknown"]))

    # Test marker sizes
    assert all(col in processed_df.columns for col in ["marker_size", "marker_symbol"])

    # Test that taxonomy columns are added
    assert all(col in processed_df.columns for col in ["species", "domain", "kingdom"])

    # Test 'selected' column default
    assert processed_df["selected"].dtype == bool
    assert all(processed_df["selected"] == False)