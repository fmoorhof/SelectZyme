from __future__ import annotations

import argparse
import logging
from time import time

import pandas as pd
import yaml

from selectzyme.backend.parsing import ParseLocalFiles


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
