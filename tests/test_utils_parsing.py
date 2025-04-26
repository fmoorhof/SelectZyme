from __future__ import annotations

from unittest.mock import mock_open, patch

import pytest
import yaml
from selectzyme.backend.utils import parse_args


def test_parse_args_valid_config():
    config_content = """
    project: 
      name: "argparse_test_minimal"
      port: 8050
      data:
        query_terms: 
          - "ec:1.13.11.85"
          - "ec:1.13.11.84"
        length: "200 TO 601"
        custom_data_location: "/raid/data/fmoorhof/PhD/SideShit/LCP/custom_seqs_no_signals.csv"
        out_dir: "datasets/output/"
        df_coi: ["accession", "reviewed", "ec", "organism_id", "length", "xref_brenda", "xref_pdb", "sequence"]
      plm:
        plm_model: "prott5"
      clustering:
        min_samples: 10
        min_cluster_size: 15
      dimred:
        random_state: 42
        method: "TSNE"
        n_neighbors: 15
    """
    with patch("builtins.open", mock_open(read_data=config_content)):
        with patch("sys.argv", ["utils.py", "--config", "test_config.yml"]):
            config = parse_args()
            assert config["project"]["name"] == "argparse_test_minimal"
            assert config["project"]["data"]["query_terms"] == [
                "ec:1.13.11.85",
                "ec:1.13.11.84",
            ]
            assert config["project"]["data"]["length"] == "200 TO 601"
            assert (
                config["project"]["data"]["custom_data_location"]
                == "/raid/data/fmoorhof/PhD/SideShit/LCP/custom_seqs_no_signals.csv"
            )
            assert config["project"]["data"]["out_dir"] == "datasets/output/"
            assert config["project"]["data"]["df_coi"] == [
                "accession",
                "reviewed",
                "ec",
                "organism_id",
                "length",
                "xref_brenda",
                "xref_pdb",
                "sequence",
            ]
            assert config["project"]["plm"]["plm_model"] == "prott5"
            assert config["project"]["clustering"]["min_samples"] == 10
            assert config["project"]["clustering"]["min_cluster_size"] == 15
            assert config["project"]["dimred"]["random_state"] == 42
            assert config["project"]["dimred"]["method"] == "TSNE"
            assert config["project"]["dimred"]["n_neighbors"] == 15


@pytest.mark.skip(reason="Minimum required fields not implemented/decided on yet")
def test_parse_args_missing_required_fields():
    config_content = """
    project: 
      name: "argparse_test_minimal"
      port: 8050
      data:
        query_terms: 
          - "ec:1.13.11.85"
          - "ec:1.13.11.84"
        length: "200 TO 601"
        custom_data_location: "/raid/data/fmoorhof/PhD/SideShit/LCP/custom_seqs_no_signals.csv"
        out_dir: "datasets/output/"
        df_coi: ["accession", "reviewed", "ec", "organism_id", "length", "xref_brenda", "xref_pdb", "sequence"]
      clustering:
        min_samples: 10
        min_cluster_size: 15
      dimred:
        random_state: 42
        method: "TSNE"
        n_neighbors: 15
    """
    with patch("builtins.open", mock_open(read_data=config_content)):
        with patch("sys.argv", ["utils.py", "--config", "test_config.yml"]):
            with pytest.raises(SystemExit):
                parse_args()


def test_parse_args_invalid_yaml():
    config_content = """
    project: "argparse_test_minimal"
      port: 8050
      data:
        query_terms: 
          - "ec:1.13.11.85"
          - "ec:1.13.11.84"
        length: "200 TO 601"
        custom_data_location: "/raid/data/fmoorhof/PhD/SideShit/LCP/custom_seqs_no_signals.csv"
        out_dir: "datasets/output/"
        df_coi: ["accession", "reviewed", "ec", "organism_id", "length", "xref_brenda", "xref_pdb", "sequence"
      plm:
        plm_model: "prott5"
      clustering:
        min_samples: 10
        min_cluster_size: 15
      dimred:
        random_state: 42
        method: "TSNE"
        n_neighbors: 15
    """
    with patch("builtins.open", mock_open(read_data=config_content)):
        with patch("sys.argv", ["utils.py", "--config", "test_config.yml"]):
            with pytest.raises(yaml.YAMLError):
                parse_args()
