from __future__ import annotations

from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from selectzyme.backend.utils import (
    export_annotated_fasta,
    export_data,
    parse_and_preprocess,
    parse_args,
    run_time,
)


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


@patch("selectzyme.backend.utils.parse_data")
@patch("selectzyme.backend.utils.Preprocessing")
@patch("selectzyme.backend.utils.custom_plotting")
def test_parse_and_preprocess(mock_custom_plotting, mock_Preprocessing, mock_parse_data):
    mock_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    mock_parse_data.return_value = mock_df
    mock_Preprocessing.return_value.preprocess.return_value = mock_df
    mock_custom_plotting.return_value = mock_df

    config = {
        "project": {
            "data": {
                "query_terms": ["query"],
                "length": "200 TO 500",
                "custom_data_location": None,
                "df_coi": None,
            },
            "preprocessing": True,
            "plot_customizations": {
                "size": 10,
                "shape": "circle",
                "marker_property": ["a", "b"],  # Move marker_property inside plot_customizations
            },
        }
    }

    result = parse_and_preprocess(config=config, existing_file=None)
    
    mock_parse_data.assert_called_once()
    mock_Preprocessing.assert_called_once()
    # Check that marker_property is passed as argument
    assert mock_custom_plotting.call_args[1].get("marker_property") == ["a", "b"]
    pd.testing.assert_frame_equal(result, mock_df)


@patch("selectzyme.backend.utils.export_annotated_fasta")
@patch("selectzyme.backend.utils.os.makedirs")
@patch("selectzyme.backend.utils.np.savez_compressed")
@patch("selectzyme.backend.utils.pd.DataFrame.to_csv")
@patch("selectzyme.backend.utils.pd.DataFrame.to_parquet")
def test_export_data(mock_to_parquet, mock_to_csv, mock_savez, mock_makedirs, mock_export_annotated_fasta):
    df = pd.DataFrame({"col1": ["val1"], "sequence": ["SEQ"]})
    X_red = np.random.rand(5, 2)
    mst = np.random.rand(5, 5)
    linkage = np.random.rand(5, 5)
    analysis_path = "mock_path"

    export_data(df, X_red, mst, linkage, analysis_path)

    mock_makedirs.assert_called_once_with(analysis_path, exist_ok=True)
    mock_to_parquet.assert_called_once()
    assert mock_to_csv.call_count == 2  # csv + tsv
    mock_savez.assert_called_once_with(
        f"{analysis_path}/x_red_mst_slc.npz", X_red=X_red, mst=mst, linkage=linkage
    )
    mock_export_annotated_fasta.assert_called_once()


def test_export_annotated_fasta(tmp_path):
    df = pd.DataFrame({
        "col1": ["header1"],
        "sequence": ["ACDEFGHIK"]
    })
    out_file = tmp_path / "test.fasta"

    export_annotated_fasta(df, str(out_file))

    content = out_file.read_text()
    assert ">header1\nACDEFGHIK\n" == content


def test_run_time_decorator():
    @run_time
    def dummy_function(x):
        return x * 2

    result = dummy_function(5)
    assert result == 10