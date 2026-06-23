import requests, time, pathlib, re

# === Configuration ===
FASTA_FILE = "tests/head_10.fasta"   # your input FASTA filename
OUT_DIR = pathlib.Path("pdbs")   # output folder for PDB files
OUT_DIR.mkdir(exist_ok=True)
API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{}"



def extract_uniprot_ids(fasta_path):
    """Extract UniProt IDs from FASTA headers (>ID|...)."""
    ids = []
    pattern = re.compile(r"^>([A-Z0-9]+)\|")
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                m = pattern.match(line)
                if m:
                    ids.append(m.group(1))
    return ids

def download_pdb(uniprot_id):
    """Download only the .pdb file for a given UniProt ID from AlphaFold DB."""
    try:
        r = requests.get(API_URL.format(uniprot_id), timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data or "pdbUrl" not in data[0]:
            print(f"[WARN] {uniprot_id}: no PDB found")
            return
        pdb_url = data[0]["pdbUrl"]
        pdb_path = OUT_DIR / f"{uniprot_id}.pdb"
        if pdb_path.exists():
            print(f"[SKIP] {pdb_path.name} already exists")
            return
        with requests.get(pdb_url, stream=True, timeout=60) as s:
            s.raise_for_status()
            with open(pdb_path, "wb") as fh:
                for chunk in s.iter_content(8192):
                    fh.write(chunk)
        print(f"[OK] Saved {pdb_path.name}")
    except Exception as e:
        print(f"[ERR] {uniprot_id}: {e}")


if __name__ == "__main__":
    ids = extract_uniprot_ids(FASTA_FILE)
    print(f"Found {len(ids)} UniProt IDs.")
    for uid in ids:
        download_pdb(uid)
        time.sleep(0.2)
