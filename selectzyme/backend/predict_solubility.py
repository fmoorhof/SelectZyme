from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import torch
from PredictionServer.data import BatchConverter, FastaBatchedDataset  # part of local netsolp install
from PredictionServer.predict import get_preds_split, sigmoid
from tqdm import tqdm

from selectzyme.backend.utils import run_time


@run_time
def get_preds(df, args, pred_frequency=2):
    alphabet_path = os.path.join(args.MODELS_PATH, "ESM1b_alphabet.pkl")  # args.MODELS_PATH

    with open(alphabet_path, "rb") as f:
        alphabet = pickle.load(f)
    #alphabet = Alphabet(proteinseq_toks)

    df.rename(columns={"accession": "sid", "sequence": "fasta"}, inplace=True)
    embed_dataset = FastaBatchedDataset(df)
    embed_batches = embed_dataset.get_batch_indices(0, extra_toks_per_seq=1)
    embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=BatchConverter(alphabet), batch_sampler=embed_batches)
    
    if args.PREDICTION_TYPE == "SU":
        preds_per_split_S = []
        preds_per_split_U = []
        for i in tqdm(range(pred_frequency)):
            pred_df_S = get_preds_split(i, embed_dataloader, args, "Solubility", df)
            preds_i_S = sigmoid(np.stack(pred_df_S.preds.to_numpy()))
            preds_per_split_S.append(preds_i_S)

            pred_df_U = get_preds_split(i, embed_dataloader, args, "Usability", df)
            preds_i_U = sigmoid(np.stack(pred_df_U.preds.to_numpy()))
            preds_per_split_U.append(preds_i_U)

        avg_pred_sol = sum(preds_per_split_S) / pred_frequency
        df["netSolP_solubility"] = pd.Series(avg_pred_sol)
        avg_pred_usa = sum(preds_per_split_U) / pred_frequency
        df["netSolP_usability"] = pd.Series(avg_pred_usa)
    elif args.PREDICTION_TYPE == "S":
        preds_per_split = []
        for i in tqdm(range(pred_frequency)):
            pred_df = get_preds_split(i, embed_dataloader, args, "Solubility", df)
            preds_i = sigmoid(np.stack(pred_df.preds.to_numpy()))
            preds_per_split.append(preds_i)
        avg_pred = sum(preds_per_split) / pred_frequency
        df["netSolP_solubility"] = pd.Series(avg_pred)
    elif args.PREDICTION_TYPE == "U":
        preds_per_split = []
        for i in tqdm(range(pred_frequency)):
            pred_df = get_preds_split(i, embed_dataloader, args, "Usability", df)
            preds_i = sigmoid(np.stack(pred_df.preds.to_numpy()))
            preds_per_split.append(preds_i)
        avg_pred = sum(preds_per_split) / pred_frequency
        df["netSolP_usability"] = pd.Series(avg_pred)
    
    df.rename(columns={"sid": "accession", "fasta": "sequence"}, inplace=True)

    return df



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

    df = ParseLocalFiles("scripts/data.fasta").parse_fasta()
    X = gen_embedding(sequences=df["sequence"].tolist(), plm_model="esm1b")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--MODEL_TYPE", default="ESM1b")
    parser.add_argument("--MODELS_PATH", default="/scratch/global_1/fmoorhof/NetSolP/models/")  # NetSolP-1.0/PredictionServer/models
    parser.add_argument("--NUM_THREADS", default=os.cpu_count(), type=int)
    parser.add_argument(
        "--PREDICTION_TYPE",
        default="SU",
        choices=['S', 'U', 'SU'],
        type=str,
        help="Either Solubility(S), Usability(U) or Both"
    )
    args = parser.parse_args()

    df_netsolp = get_preds(df=df, args=args)  # , out_path="solubility.csv")
    print(df_netsolp)
    