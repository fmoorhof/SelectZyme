import pathlib
import re
import time

import requests

# ===========================
# Configuration
# ===========================
FASTA_FILE = "tests/head_10.fasta"
OUT_DIR = pathlib.Path("pdbs")
OUT_DIR.mkdir(exist_ok=True)

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{}"
ESM_API = "https://api.esmatlas.com/fetchPredictedStructure/{}"


def extract_ids(fasta_path):
    """
    Extract identifiers from FASTA headers.

    Expected formats:
        >P12345|...
        >A0A023GPI8|...
        >MGYP002537940442|...
    """
    ids = []

    pattern = re.compile(r"^>([^|\s]+)")

    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                m = pattern.match(line)
                if m:
                    ids.append(m.group(1))

    return ids


def download_from_alphafold(uniprot_id):
    """Download a PDB from AlphaFoldDB."""

    pdb_path = OUT_DIR / f"{uniprot_id}.pdb"

    if pdb_path.exists():
        print(f"[SKIP] {pdb_path.name} already exists")
        return

    r = requests.get(ALPHAFOLD_API.format(uniprot_id), timeout=15)
    r.raise_for_status()

    data = r.json()

    if not data or "pdbUrl" not in data[0]:
        print(f"[WARN] {uniprot_id}: no AlphaFold model found")
        return

    pdb_url = data[0]["pdbUrl"]

    with requests.get(pdb_url, stream=True, timeout=60) as s:
        s.raise_for_status()

        with open(pdb_path, "wb") as fh:
            for chunk in s.iter_content(8192):
                fh.write(chunk)

    print(f"[AFDB] Saved {pdb_path.name}")


def download_from_esm(mgyp_id):
    """Download a PDB directly from the ESM Metagenomic Atlas."""

    pdb_path = OUT_DIR / f"{mgyp_id}.pdb"

    if pdb_path.exists():
        print(f"[SKIP] {pdb_path.name} already exists")
        return

    url = ESM_API.format(mgyp_id)

    r = requests.get(url, timeout=60)

    if r.status_code == 404:
        print(f"[WARN] {mgyp_id}: no ESM model found")
        return

    r.raise_for_status()

    with open(pdb_path, "w") as fh:
        fh.write(r.text)

    print(f"[ESM ] Saved {pdb_path.name}")


def download_pdb(identifier):
    """
    Automatically choose the correct database.

    MGYP*  -> ESM Metagenomic Atlas
    others -> AlphaFoldDB
    """

    try:
        if identifier.upper().startswith("MGYP"):
            download_from_esm(identifier)
        else:
            download_from_alphafold(identifier)

    except Exception as e:
        print(f"[ERR] {identifier}: {e}")


if __name__ == "__main__":

    ids = extract_ids(FASTA_FILE)

    print(f"Found {len(ids)} identifiers.")

    for identifier in ids:
        download_pdb(identifier)
        time.sleep(0.2)
