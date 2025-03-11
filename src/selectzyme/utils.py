from __future__ import annotations

import argparse
import logging
import os
from time import time

import pandas as pd
import yaml

from selectzyme.fetch_data_uniprot import UniProtFetcher
from selectzyme.parsing import Parsing


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


def parse_data(project_name, query_terms, length, custom_file, out_dir, df_coi):
    exisiting_file = out_dir + project_name + ".tsv"
    df = _parse_data(exisiting_file, custom_file, query_terms, length, df_coi)
    df = _clean_data(df)
    return df


def _parse_data(
    exisiting_file: str, custom_file: str, query_terms: list, length: str, df_coi: list
):
    if os.path.isfile(exisiting_file):
        return Parsing(exisiting_file).parse()

    if (
        query_terms != [""]
    ):  # todo: handle if query_terms NoneType -> breaks execution when query_terms not defined in config.yml
        fetcher = UniProtFetcher(df_coi)
    if custom_file != "":
        df_custom = Parsing(custom_file).parse()
        if query_terms == [""]:
            return df_custom
        df = fetcher.query_uniprot(query_terms, length)
        df = pd.concat(
            [df_custom, df], ignore_index=True
        )  # custom data first that they are displayed first in plot legends
        return df
    elif query_terms != [""]:
        return fetcher.query_uniprot(query_terms, length)
    else:
        raise ValueError(
            "No query terms or custom data location provided. Please provide either one."
        )

def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        df["accession"] != "Entry"
    ]  # remove concatenated headers that are introduced by each query term
    logging.info(f"Total amount of retrieved entries: {df.shape[0]}")
    df.drop_duplicates(subset="accession", keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logging.info(f"Total amount of non redundant entries: {df.shape[0]}")
    if "xref_brenda" in df.columns:
        logging.info(
            f"Amount of BRENDA reviewed entries: {df['xref_brenda'].notna().sum()}"
        )
    return df


def export_annotated_fasta(df: pd.DataFrame, out_file: str):
    with open(out_file, "w") as f_out:
        for index, row in df.iterrows():
            fasta = (
                ">",
                "|".join(row.iloc[:-1].map(str)),
                "\n",
                row.loc["sequence"],
                "\n",
            )  # todo: exclude seq in header (-1=last column potentially broken)
            f_out.writelines(fasta)
    logging.info(f"FASTA file written to {out_file}")


@DeprecationWarning
def convert_tabular_to_fasta(in_file: str, out_file: str):
    df = Parsing(in_file).parse()
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
