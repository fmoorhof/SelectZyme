import logging

from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_embedding(
    sequences: list[str], plm_model: str = "esm1b", no_pad: bool = False
) -> np.ndarray:
    """
    Generate embeddings for a list of sequences using a specified pre-trained language model (PLM).
    Args:
        sequences (list[str]): A list of sequences to generate embeddings for.
        plm_model (str, optional): The pre-trained language model to use. Default is 'esm1b'.
                                   Supported models include 'esm1b', 'prott5', and 'prostt5'.
        no_pad (bool, optional): If True, removes paddings from the sequences before generating mean embeddings.
                                 Default is False.
    Returns:
        np.ndarray: An array of embeddings for the input sequences.
    """

    tokenizer, model = _select_plm_model(plm_model)
    logging.info(f"Generating {plm_model} embeddings using device: {device}")

    if (
        plm_model == "prott5" or plm_model == "prostt5"
    ):  # models require sequenes to be spaced (whitespace)
        sequences = [" ".join(list(seq)) for seq in sequences]

    embeddings = []
    for seq in tqdm(sequences):
        inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(**inputs)

        if no_pad == False:
            last_hidden_states = outputs.last_hidden_state
            embedding = (
                torch.mean(last_hidden_states, dim=1).squeeze().cpu().numpy()
            )  # Move embedding back to CPU
        else:
            if plm_model == "prott5" or plm_model == "prostt5":
                seq_len = int(len(seq) / 2 + 1)
            else:
                seq_len = len(seq)
            last_hidden_states = outputs.last_hidden_state[
                0, :seq_len, :
            ]  # Remove padding
            embedding = (
                last_hidden_states.mean(dim=0).cpu().numpy()
            )  # Move embedding back to CPU
        embeddings.append(embedding)

    # Free up GPU memory (somehow not done by default)
    torch.cuda.empty_cache()

    return np.array(embeddings)


def _select_plm_model(plm_model: str = "esm1b") -> tuple:
    """
    Selects and loads a pre-trained language model (PLM) and its corresponding tokenizer based on the specified model name.
    Args:
        plm_model (str): The name of the pre-trained language model to load.
                         Options are 'esm1b' (default), 'esm2', 'prott5', 'prostt5'.
    Returns:
        tuple: A tuple containing the tokenizer and the model.
    Raises:
        ValueError: If the specified model name is not supported.
        NotImplementedError: If 'esm3' is specified, as it is not yet implemented.
    """

    if plm_model == "esm1b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
        model = AutoModel.from_pretrained("facebook/esm1b_t33_650M_UR50S").to(device)

    elif plm_model == "esm2":
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)

    elif plm_model == "prott5":
        from transformers import T5Tokenizer, T5EncoderModel

        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)

    elif plm_model == "prostt5":
        from transformers import T5Tokenizer, T5EncoderModel

        tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5")
        model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)

    else:
        raise ValueError(
            f"Model {plm_model} not supported. Please choose one of: 'esm1b', 'esm2', 'prott5', 'prostt5'"
        )

    return tokenizer, model


if __name__ == "__main__":
    # load example data
    from parsing import Parsing
    from preprocessing import Preprocessing

    df = Parsing("tests/head_10.tsv").parse_tsv()
    # df = Parsing('datasets/output/petase.tsv').parse_tsv()
    # df = Preprocessing(df).preprocess()
    pp = Preprocessing(df)
    # pp.remove_long_sequences()
    # pp.remove_sequences_without_Metheonin()
    # pp.remove_sequences_with_undertermined_amino_acids()
    pp.remove_duplicate_entries()
    pp.remove_duplicate_sequences()
    df = pp.df

    # test embedding
    embeddings = gen_embedding(
        df["sequence"].tolist(), plm_model="esm2"
    )  # , no_pad=True)
    print(embeddings.shape)
    print(embeddings)

    # # quickly generate embeddings
    # from utils import database_access
    # database_access(df, project_name='petase', plm_model='prott5')
