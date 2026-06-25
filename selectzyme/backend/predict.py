from __future__ import annotations

import logging
import os
from types import SimpleNamespace


def run_predictions(df, X, config, analysis_path):

    # predict EC numbers with CLEAN if they dont exist yet
    if df.CLEAN_EC_pred.isnull().all() and df.CLEAN_probability.isnull().all():
        if config["project"]["plm"]["plm_model"] == "esm1b" and config["project"]["predictions"]["ec"] == True:
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
                device="cpu",
            )
        
            df = df.merge(df_clean[["accession", "CLEAN_EC_pred", "CLEAN_probability"]], on="accession", how="left")

    # predict solubility and usability with NetSolP if they dont exist yet
    if config["project"]["predictions"]["solubility"] == True and df.netSolP_solubility.isnull().all() and df.netSolP_usability.isnull().all():
        from selectzyme.backend.predict_solubility import get_preds
        logging.info("Running NetSolP predictions. This might take a long time.")

        # hard coded configurations to pass to NetSolP package
        args = SimpleNamespace(
            MODEL_TYPE="ESM1b",
            MODELS_PATH="NetSolP-1.0/PredictionServer/models",  # /scratch/global_1/fmoorhof/NetSolP/models/
            NUM_THREADS=os.cpu_count(),
            PREDICTION_TYPE="SU",  # Either Solubility(S), Usability(U) or Both
        )

        df = get_preds(df=df, args=args)
    
    return df