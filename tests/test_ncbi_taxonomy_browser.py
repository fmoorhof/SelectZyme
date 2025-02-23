import pytest
from src.ncbi_taxonomy_resolver import lineage_resolver


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
