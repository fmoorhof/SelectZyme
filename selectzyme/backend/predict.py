from __future__ import annotations

import logging
import os
from numbers import Real
from types import SimpleNamespace

import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weighted_prioritization(df, config):
    """
    Calculate a weighted prioritization score for each sequence using
    median-thresholded predictions and weights defined in the config.

    Args:
        df (pd.DataFrame): DataFrame containing the predictions.
        config (dict): Configuration dictionary containing the weights for each prediction.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'weighted_priority' containing the calculated scores.
    """
    # Initialize the weighted priority column
    df["weighted_priority"] = 0.0

    # Define the prediction columns and their corresponding weights from the config
    prediction_cfg = config.get("project", {}).get("predictions", {})
    prediction_columns = {
        "CLEAN_probability": prediction_cfg.get("ec", {}).get("weight"),
        "netSolP_solubility": prediction_cfg.get("solubility", {}).get("weight"),
        "netSolP_usability": prediction_cfg.get("usability", {}).get("weight"),
    }

    def _parse_prediction_value(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, Real):
            return float(value)
        if isinstance(value, str):
            candidates = [part.strip() for part in value.split(";")]
            numeric_candidates = pd.to_numeric(candidates, errors="coerce")
            valid_candidates = numeric_candidates[~pd.isna(numeric_candidates)]
            if len(valid_candidates) > 0:
                return float(valid_candidates.max())
        return pd.NA

    # Calculate the weighted priority using per-column median thresholds
    for col, weight in prediction_columns.items():
        if col not in df.columns:
            continue
        if isinstance(weight, bool) or not isinstance(weight, Real):
            continue

        numeric_col = df[col].apply(_parse_prediction_value)
        median_value = numeric_col.median()
        above_median = (numeric_col > median_value).fillna(False).astype(float)
        df["weighted_priority"] += above_median * weight

    return df


def run_predictions(df, X, config, analysis_path):

    # predict EC numbers with CLEAN if they dont exist yet
    if "CLEAN_EC_pred" not in df.columns and "CLEAN_probability" not in df.columns:
        if config["project"]["plm"]["plm_model"] == "esm1b" and config["project"]["predictions"]["ec"]["enabled"] == True:
            from selectzyme.backend.predict_ec import run_clean_inference_with_embeddings

            data_path = "../CLEAN/app/data/"
            desired_split = "100"
            df_clean = run_clean_inference_with_embeddings(
                sequence_label_esm_emb_dict={row["accession"]: X[i] for i, row in df.iterrows()},
                emb_train_path=f'{data_path}pretrained/{desired_split}.pt',
                ec_csv_path=f'{data_path}split{desired_split}.csv',
                model_ckpt_path=f'{data_path}pretrained/split{desired_split}.pth',
                out_csv=analysis_path + "/clean_predictions.csv",
                gmm=f'{data_path}pretrained/gmm_ensumble.pkl',
                device=device,
            )
        
            df = df.merge(df_clean[["accession", "CLEAN_EC_pred", "CLEAN_probability"]], on="accession", how="left")

        else:
            logging.info("CLEAN predictions are only available for ESM1b embeddings. Skipping prediction. Other reasons can be column names CLEAN_EC_pred and CLEAN_probability already exist in the dataframe or config['project']['predictions']['ec'] is set to False.")

    # predict solubility with NetSolP if they dont exist yet
    if config["project"]["predictions"]["solubility"]["enabled"] == True and "netSolP_solubility" not in df.columns:
        from selectzyme.backend.predict_solubility import get_preds
        logging.info("Running NetSolP predictions. This might take a long time.")

        # hard coded configurations to pass to NetSolP package
        args = SimpleNamespace(
            MODEL_TYPE="ESM1b",
            MODELS_PATH="/scratch/global_1/fmoorhof/NetSolP/models/",  # NetSolP-1.0/PredictionServer/models
            NUM_THREADS=os.cpu_count(),
            PREDICTION_TYPE="S",
        )

        df = get_preds(df=df, args=args)

    else:
        logging.info("NetSolP predictions already exist (columns named netSolP_solubility and netSolP_usability). Skipping prediction.")

    # predict usability with NetSolP if they dont exist yet
    if config["project"]["predictions"]["usability"]["enabled"] == True and "netSolP_usability" not in df.columns:
        from selectzyme.backend.predict_solubility import get_preds
        logging.info("Running NetSolP predictions. This might take a long time.")

        # hard coded configurations to pass to NetSolP package
        args = SimpleNamespace(
            MODEL_TYPE="ESM1b",
            MODELS_PATH="/scratch/global_1/fmoorhof/NetSolP/models/",  # NetSolP-1.0/PredictionServer/models
            NUM_THREADS=os.cpu_count(),
            PREDICTION_TYPE="U",
        )

        df = get_preds(df=df, args=args)

    else:
        logging.info("NetSolP predictions already exist (columns named netSolP_solubility and netSolP_usability). Skipping prediction.")
    
    df = weighted_prioritization(df, config)
    
    return df