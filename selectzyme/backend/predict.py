import os
import logging
import argparse

from CLEAN.distance_map import get_dist_map_test
from CLEAN.evaluate import infer_confidence_gmm, maximum_separation
from CLEAN.model import LayerNormNet
from CLEAN.utils import get_ec_id_dict

from selectzyme.backend.predict_ec import run_clean_inference_with_embeddings
from selectzyme.backend.predict_solubility import get_preds



def run_predictions(df, X, config, analysis_path):

    # predict EC numbers with CLEAN if they dont exist yet
    if df.CLEAN_EC_pred.isnull().all() and df.CLEAN_probability.isnull().all():
        if config["project"]["plm"]["plm_model"] == "esm1b" and config["project"]["predictions"]["ec"] == True:
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
        logging.info("Running NetSolP predictions. This might take a long time.")
        # hard coded configurations to pass to NetSolP package
        parser = argparse.ArgumentParser()
        parser.add_argument("--MODEL_TYPE", default="ESM1b")
        parser.add_argument("--MODELS_PATH", 
                            default="/scratch/global_1/fmoorhof/NetSolP/models/")  # NetSolP-1.0/PredictionServer/models
        parser.add_argument("--NUM_THREADS", default=os.cpu_count(), type=int)
        parser.add_argument(
            "--PREDICTION_TYPE",
            default="SU",
            choices=['S', 'U', 'SU'],
            type=str,
            help="Either Solubility(S), Usability(U) or Both"
        )
        args = parser.parse_args()

        df = get_preds(df=df, args=args)
    
    return df