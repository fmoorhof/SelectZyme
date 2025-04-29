from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_embeddings(df: pd.DataFrame, 
                    plm_model: str, 
                    embedding_file: str) -> np.ndarray:
    """
    Load or generate embeddings for a given dataset using a specified pre-trained language model (PLM).
    If the embedding file exists, the function loads the embeddings from the file.
    Otherwise, it generates embeddings using the provided PLM model and saves them to the file.
    Args:
        df (pd.DataFrame): A pandas DataFrame containing a column "sequence" with the input sequences.
        plm_model (str): The name or identifier of the pre-trained language model to use for generating embeddings.
        embedding_file (str): The file path to load/save the embeddings.
    Returns:
        np.ndarray: A NumPy array containing the embeddings.
    """
    if os.path.exists(embedding_file):
        X = np.load(embedding_file)["X"]
        logging.info(f"Loaded embeddings from {embedding_file}")
    else:
        X = gen_embedding(
            sequences=df["sequence"].tolist(),
            plm_model=plm_model,
        )
        np.savez_compressed(embedding_file, X=X)
        logging.info(f"Saved embeddings to {embedding_file}")
    return X


def gen_embedding(
    sequences: list[str], plm_model: str = "esm1b", no_pad: bool = False
) -> np.ndarray:
    """
    Generate embeddings for a list of sequences using a specified pre-trained language model (PLM).

    Args:
        sequences (list[str]): List of amino acid sequences.
        plm_model (str, optional): Pre-trained model name. Options: 'esm1b', 'esm2', 'prott5', 'prostt5'.
        no_pad (bool, optional): If True, removes padding tokens when calculating mean embedding.

    Returns:
        np.ndarray: Array of embeddings.
    """
    tokenizer, model = _load_model_and_tokenizer(plm_model)
    logging.info(f"Generating embeddings with {plm_model} on device: {device}")

    formatted_sequences = _format_sequences(sequences, plm_model)

    embeddings = [_generate_sequence_embedding(seq, tokenizer, model, plm_model, no_pad) for seq in tqdm(formatted_sequences)]

    torch.cuda.empty_cache()
    return np.array(embeddings)


def _load_model_and_tokenizer(plm_model: str) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Load the tokenizer and model for a given PLM."""
    if plm_model == "esm1b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
        model = AutoModel.from_pretrained("facebook/esm1b_t33_650M_UR50S").to(device)

    elif plm_model == "esm2":
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)

    elif plm_model == "prott5":
        from transformers import T5EncoderModel, T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)

    elif plm_model == "prostt5":
        from transformers import T5EncoderModel, T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5")
        model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)

    else:
        raise ValueError(
            f"Unsupported model '{plm_model}'. Choose from 'esm1b', 'esm2', 'prott5', 'prostt5'."
        )

    return tokenizer, model


def _format_sequences(sequences: list[str], plm_model: str) -> list[str]:
    """Format sequences if necessary (e.g., insert spaces for T5 models)."""
    if plm_model in {"prott5", "prostt5"}:
        return [" ".join(list(seq)) for seq in sequences]
    return sequences


def _generate_sequence_embedding(
    sequence: str,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    plm_model: str,
    no_pad: bool,
) -> np.ndarray:
    """Generate embedding for a single sequence."""
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    if no_pad:
        return _extract_no_pad_embedding(outputs, sequence, plm_model)
    else:
        return _extract_mean_embedding(outputs, sequence, plm_model)


def _extract_mean_embedding(
    outputs: torch.nn.Module,
    sequence: str,
    plm_model: str,
) -> np.ndarray:
    """Extract mean embedding including padding."""
    try:
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    except RuntimeError as e:
        if plm_model == "esm1b":
            raise RuntimeError(
                f"ESM-1b model cannot handle sequences longer than 1024 amino acids.\n"
                f"Problematic sequence: {sequence}\n"
                "Please filter or truncate long sequences or use 'prott5' instead."
            ) from e
        raise
    return embedding


def _extract_no_pad_embedding(
    outputs: torch.nn.Module,
    sequence: str,
    plm_model: str,
) -> np.ndarray:
    """Extract mean embedding after removing padding."""
    seq_len = len(sequence) if plm_model not in {"prott5", "prostt5"} else int(len(sequence) / 2 + 1)
    return outputs.last_hidden_state[0, :seq_len, :].mean(dim=0).cpu().numpy()



if __name__ == "__main__":
    # load example data
    from parsing import ParseLocalFiles
    from preprocessing import Preprocessing

    df = ParseLocalFiles("tests/head_10.tsv").parse_tsv()
    # df = Parsing('datasets/output/ired.tsv').parse_tsv()
    # df = Preprocessing(df).preprocess()
    pp = Preprocessing(df)
    pp.remove_long_sequences()
    # pp.remove_sequences_without_Metheonin()
    # pp.remove_sequences_with_undertermined_amino_acids()
    pp.remove_duplicate_entries()
    pp.remove_duplicate_sequences()
    df = pp.df

    # test embedding
    embeddings = gen_embedding(
        df["sequence"].tolist(), plm_model="prostt5"
    )  # , no_pad=True)
    print(embeddings.shape)
    print(embeddings)

    # # quickly generate embeddings
    # from utils import database_access
    # database_access(df, project_name='petase', plm_model='prott5')
