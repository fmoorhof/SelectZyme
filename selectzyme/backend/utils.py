from __future__ import annotations

import argparse
import logging
import os
from time import time

import numpy as np
import pandas as pd
import yaml

from selectzyme.backend.customizations import custom_plotting
from selectzyme.backend.parsing import ParseLocalFiles, parse_data
from selectzyme.backend.preprocessing import Preprocessing


def parse_args():
    parser = argparse.ArgumentParser(description="Process parameters.")

    parser.add_argument(
        "--config",
        type=str,
        default="results/input_configs/test_config.yml",
        help="Path to a config.yml file with all parameters (default location: results/input_configs/test_config.yml)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Validate required arguments
    if not all([config["project"]["data"], config["project"]["plm"]]):
        parser.error(
            "Some required arguments, either CLI or from config.yml, not provided."
        )

    return config


def parse_and_preprocess(config, existing_file) -> pd.DataFrame:
    """
    Parse and preprocess data based on the provided configuration (config.yml).
    This function parses data using the configuration parameters, applies 
    preprocessing if specified, and customizes the data for plotting.
    Returns:
        pandas.DataFrame: The processed and customized DataFrame.
    """
    df = parse_data(
        config["project"]["data"]["query_terms"],
        config["project"]["data"]["length"],
        config["project"]["data"]["custom_data_location"],
        existing_file,
        config["project"]["data"]["df_coi"],
    )

    if config["project"]["preprocessing"]:
        df = Preprocessing(df).preprocess()

    # Apply customizations
    df = custom_plotting(df=df, 
                         size=config["project"]["plot_customizations"]["size"], 
                         shape=config["project"]["plot_customizations"]["shape"])
    return df


def export_data(df: pd.DataFrame, 
                X_red: np.ndarray, 
                mst_array: np.ndarray, 
                linkage_array: np.ndarray,
                analysis_path: str) -> None:
    """
    Exports various data structures to files in specified formats for further use.
    """
    def sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares a Pandas DataFrame for saving as a Parquet file by sanitizing its columns.
        This function converts all columns with the data type 'object' to 'string' to avoid 
        potential type inference issues when using pyarrow for Parquet serialization.
        Args:
            df (pd.DataFrame): The input DataFrame to be sanitized.
        Returns:
            pd.DataFrame: A copy of the input DataFrame with 'object' columns converted to 'string'.
        """
        obj_cols = df.select_dtypes(include=["object"]).columns
        return df.copy().astype({col: "string" for col in obj_cols})

    # export data for minimal front end
    os.makedirs(analysis_path, exist_ok=True)
    df_export = sanitize_for_parquet(df)
    df_export.to_parquet(os.path.join(analysis_path, "df.parquet"), index=False)
    np.savez_compressed(os.path.join(analysis_path, "x_red_mst_slc.npz"),
                        X_red=X_red,
                        mst=mst_array,
                        linkage=linkage_array)
    
    # export data for user
    df.to_csv(os.path.join(analysis_path + "/data.csv"), index=False)
    df.to_csv(analysis_path + "/data.tsv", sep="\t", index=False)
    export_annotated_fasta(df=df, out_file=os.path.join(analysis_path + "/data.fasta"))


def export_annotated_fasta(df: pd.DataFrame, out_file: str):
    with open(out_file, "w") as f_out:
        for index, row in df.iterrows():
            header = "|".join(row.drop("sequence").map(str))
            fasta = (
                ">",
                header,
                "\n",
                row.loc["sequence"],
                "\n",
            )
            f_out.writelines(fasta)
    logging.info(f"FASTA file written to {out_file}")


@DeprecationWarning
def convert_tabular_to_fasta(in_file: str, out_file: str):
    df = ParseLocalFiles(in_file).parse()
    with open(out_file, "w") as f_out:
        for index, row in df.iterrows():
            fasta = ">", str(row.iloc[0]), "\n", row.loc["sequence"], "\n"
            f_out.writelines(fasta)
    logging.info(f"FASTA file written to {out_file}")


def run_time(func):
    """
    A decorator that measures the execution time of a function.

    Args:
        func: The function to be decorated.

    Returns:
        The wrapped function.

    """

    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(
            f"Execution time of the function {func.__name__}: {end_time - start_time} seconds"
        )
        return result

    return wrapper


if __name__ == "__main__":
    convert_tabular_to_fasta(
        "results/datasets/pet_active_region.csv",
        "results/datasets/pet_active_region.fasta",
    )