from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import torch
from PredictionServer.data import BatchConverter, FastaBatchedDataset  # part of local netsolp install
from PredictionServer.predict import get_preds_split, sigmoid


def get_preds(df, prediction_type, args, out_path):
    alphabet_path = os.path.join(args.MODELS_PATH, "ESM1b_alphabet.pkl")

    with open(alphabet_path, "rb") as f:
        alphabet = pickle.load(f)
    #alphabet = Alphabet(proteinseq_toks)
    embed_dataset = FastaBatchedDataset(df)
    embed_batches = embed_dataset.get_batch_indices(0, extra_toks_per_seq=1)
    embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=BatchConverter(alphabet), batch_sampler=embed_batches)
    
    if "S" in prediction_type:
        print("Doing Solubility")
        preds_per_split = []
        for i in range(5):
            print(f"Model {i}")
            pred_df = get_preds_split(i, embed_dataloader, args, "Solubility", df)
            preds_i = sigmoid(np.stack(pred_df.preds.to_numpy()))
            preds_per_split.append(preds_i)
            df[f"predicted_solubility_model_{i}"] = preds_i
        avg_pred = sum(preds_per_split) / 5
        df["predicted_solubility"] = pd.Series(avg_pred)
    if "U" in prediction_type:
        print("Doing Usability")
        preds_per_split = []
        for i in range(5):
            print(f"Model {i}")
            pred_df = get_preds_split(i, embed_dataloader, args, "Usability", df)
            preds_i = sigmoid(np.stack(pred_df.preds.to_numpy()))
            preds_per_split.append(preds_i)
            df[f"predicted_usability_model_{i}"] = preds_i
        avg_pred = sum(preds_per_split) / 5
        df["predicted_usability"] = pd.Series(avg_pred)

    df.to_csv(out_path, index=False)



if __name__ == "__main__":
    # mock some sequence embeddings for testing
    import sys
    from pathlib import Path

    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / "selectzyme").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            break

    from selectzyme.backend.embed import gen_embedding
    from selectzyme.backend.parsing import ParseLocalFiles

    df = ParseLocalFiles("scripts/2_old+new.fasta").parse_fasta()
    X = gen_embedding(sequences=df["sequence"].tolist(), plm_model="esm1b")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--MODEL_TYPE", default="ESM1b")
    parser.add_argument("--MODELS_PATH", default="NetSolP-1.0/PredictionServer/models")
    args = parser.parse_args()

    get_preds(df=df, prediction_type="SU", args=args, out_path="solubility.csv")
    