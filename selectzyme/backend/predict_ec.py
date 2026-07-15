from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from CLEAN.distance_map import get_dist_map_test
from CLEAN.evaluate import infer_confidence_gmm, maximum_separation
from CLEAN.model import LayerNormNet
from CLEAN.utils import get_ec_id_dict

logger = logging.getLogger(__name__)


def get_max_sep_predictions_dict(inference_df, gmm):
    """Convert a CLEAN distance matrix into ranked EC predictions.

    CLEAN produces a distance table where each column corresponds to an
    inference sequence and each row index corresponds to an EC label. This
    helper selects the 10 closest EC candidates per sequence, applies the
    maximum-separation rule to keep only the most plausible prefix of those
    candidates, and formats the result as strings like ``EC:1.1.1.1/0.1234``.

    Parameters
    ----------
    inference_df : pandas.DataFrame
        Distance matrix returned by CLEAN, with sequence IDs as columns and EC
        labels as row indices.
    gmm : str
        Path to a Gaussian mixture model used to transform raw
        distances into confidence scores.

    Returns
    -------
    dict[str, list[str]]
        Mapping from sequence ID to a list of EC predictions ordered from best
        to worst.
    """
    gmm_lst = pickle.load(open(gmm, 'rb'))

    max_sep_predictions = {}
    for sequence_label in inference_df.columns:
        smallest_10_dist_df = inference_df[sequence_label].nsmallest(10)
        dist_lst = list(smallest_10_dist_df)
        max_sep_i = maximum_separation(dist_lst, True, False)
        ec = []
        for i in range(max_sep_i+1):
            EC_i = smallest_10_dist_df.index[i]
            dist_i = smallest_10_dist_df[i]
            dist_i = infer_confidence_gmm(dist_i, gmm_lst)
            dist_str = "{:.4f}".format(dist_i)
            ec.append('EC:' + str(EC_i) + '/' + dist_str)
        max_sep_predictions[sequence_label] = ec
    return max_sep_predictions


def clean_max_sep_predictions(CLEAN_model, sequence_label_esm_emb_dict, emb_train, ec_id_dict_train, gmm_path, device):
    """Run CLEAN inference on precomputed ESM embeddings.

    This is the in-process inference step from CLEAN's original workflow. It
    avoids recomputing ESM1b embeddings by taking your already-generated
    embedding dictionary, projecting it through the CLEAN model, computing the
    distance map against CLEAN's training embeddings, and converting that map
    into final EC predictions with the maximum-separation heuristic.

    Parameters
    ----------
    CLEAN_model : torch.nn.Module
        Loaded CLEAN model used to transform ESM embeddings.
    sequence_label_esm_emb_dict : dict[str, torch.Tensor]
        Mapping from sequence ID to its precomputed ESM1b embedding.
    emb_train : torch.Tensor
        CLEAN's cached training embeddings (typically ``100.pt`` or similar).
    ec_id_dict_train : dict
        EC-to-index mapping loaded from the CLEAN training CSV.
    gmm_path : str
        Path to a Gaussian mixture model used to transform raw
        distances into confidence scores.
    device : str
        Torch device to run the CLEAN model on.

    Returns
    -------
    dict[str, list[str]]
        Mapping from sequence ID to formatted EC predictions.
    """
    esm_emb_inference = torch.cat(
        [torch.from_numpy(sequence_label_esm_emb_dict[label]).unsqueeze(0) if isinstance(sequence_label_esm_emb_dict[label], np.ndarray) else sequence_label_esm_emb_dict[label].unsqueeze(0)
            for label in sequence_label_esm_emb_dict])
    id_ec_inference_dummy = {seq_label:[] for seq_label in sequence_label_esm_emb_dict}
    with torch.no_grad():
        model_emb_inference  = CLEAN_model(esm_emb_inference.to(device))
    inference_dist = get_dist_map_test(
        emb_train, model_emb_inference, ec_id_dict_train, id_ec_inference_dummy, device, torch.float32)
    inference_df = pd.DataFrame.from_dict(inference_dist)
    max_sep_predictions_dict = get_max_sep_predictions_dict(inference_df, gmm_path)
    return max_sep_predictions_dict


def run_clean_inference_with_embeddings(
    sequence_label_esm_emb_dict: Dict[str, torch.Tensor],
    emb_train_path: str,
    ec_csv_path: str,
    model_ckpt_path: str,
    out_csv: Optional[str] = None,
    device = "cpu",
    gmm: str = "",
) -> pd.DataFrame:
    """Run CLEAN inference using already-computed ESM1b embeddings.

    Parameters:
    - sequence_label_esm_emb_dict: mapping from sequence id/label -> torch.Tensor (embedding)
    - emb_train_path: path to `100.pt` (emb_train) used by CLEAN training
    - ec_csv_path: path to the CSV used to build EC id mapping (train csv)
    - model_ckpt_path: path to the CLEAN model checkpoint (.pth)
    - out_csv: optional path to write predictions CSV
    - device: 'cpu' or 'cuda'
    - gmm: path to the Gaussian mixture model used to transform raw distances into confidence scores
    Returns a pandas.DataFrame with columns `Seq_ID` and `Prediction`.
    """
    # build model and load checkpoint
    CLEAN_model = LayerNormNet(512, 128, device, torch.float32)
    checkpoint = torch.load(model_ckpt_path, map_location=device)
    CLEAN_model.load_state_dict(checkpoint)
    CLEAN_model.eval()

    # load emb_train
    emb_train = torch.load(emb_train_path, map_location=device)

    # load ec id mapping
    _, ec_id_dict_train = get_ec_id_dict(ec_csv_path)

    # call CLEAN inference routine
    preds = clean_max_sep_predictions(
        CLEAN_model, sequence_label_esm_emb_dict, emb_train, ec_id_dict_train, gmm, device
    )

    # format to DataFrame: split each prediction "EC:3.1.1.1/0.0230" into
    # CLEAN_EC_pred -> "3.1.1.1" and CLEAN_probability -> "0.0230".
    records = []
    for k, v in preds.items():
        ecs = []
        probs = []
        for item in v:
            parts = item.split("/", 1)
            left = parts[0]
            if left.startswith("EC:"):
                left = left[len("EC:"):]
            prob = parts[1] if len(parts) > 1 else ""
            ecs.append(left)
            probs.append(prob)
        records.append({
            "accession": k,
            "CLEAN_EC_pred": "; ".join(ecs),
            "CLEAN_probability": "; ".join(probs),
        })

    df = pd.DataFrame(records)

    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        df.to_csv(out_csv, index=False)

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

    df = ParseLocalFiles("scripts/2_old+new.fasta").parse_fasta()
    X = gen_embedding(sequences=df["sequence"].tolist(), plm_model="esm1b")
    emb = {row["accession"]: X[i] for i, row in df.iterrows()}

    # Example usage
    pretraining_path = "../CLEAN/app/data/pretrained/"
    ec_csv_path = "../CLEAN/app/data/"
    desired_split = "100"
    out_csv = "clean_predictions.csv"

    df_clean = run_clean_inference_with_embeddings(
        sequence_label_esm_emb_dict=emb,
        emb_train_path=f'{pretraining_path}{desired_split}.pt',
        ec_csv_path=f'{ec_csv_path}split{desired_split}.csv',
        model_ckpt_path=f'{pretraining_path}split{desired_split}.pth',
        out_csv=out_csv,
        gmm=f'{pretraining_path}gmm_ensumble.pkl',
        device="cpu",
    )

    
    print(f"Wrote predictions to: {os.path.abspath(out_csv)}")
    df = df.merge(df_clean[["accession", "CLEAN_EC_pred", "CLEAN_probability"]], on="accession", how="left")
