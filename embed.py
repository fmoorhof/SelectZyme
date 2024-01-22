"""
This file provides basic functionalites like file parsing and esm embedding.
"""
import logging

import numpy as np
import torch
import esm 

from preprocessing import read_fasta


def read_fasta(filepath):
    """
    Iteratively returns the entries from a fasta file.
    
    Parameters
    ----------
    file : str
        the fasta file
    
    Yields
    ------
    output : tuple
        a single entry in the fasta file (header, sequence)
    """
    is_header = lambda x: x.startswith('>')
    compress  = lambda x: ''.join(_.strip() for _ in x)
    reader    = iter(groupby(open(filepath), is_header))
    reader    = iter(groupby(open(filepath), is_header)) if next(reader)[0] else reader
    for key, group in reader:
        if key:
            for header in group:
                header = header[1:].strip()
        else:
            sequence = compress(group)
            if sequence != '':
                yield header, sequence


def gen_embedding(sequences, device: str = 'cuda'):
    """
    Generate embeddings for a list of protein sequences.

    :param sequences: list containing protein sequences to embed here
    :param device: device for running the model (either cpu or gpu=cuda)
    """
    # load the esm-1b protein language model
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    model.eval()  # disable dropout for deterministic results
    model = model.to(device)

    embeddings = []

    with torch.no_grad():
        for n, s in enumerate(sequences):
            logging.info(f'Progress : {n+1} / {len(sequences)}\r')
            
            batch_labels, batch_strs, batch_tokens = batch_converter([[None, s]])
            batch_tokens = batch_tokens.to(device)
            
            # generate the full size embedding vector
            result = model(batch_tokens, repr_layers=[33], return_contacts=False)
            full_size = result["representations"][33].to('cpu')[0]
            
            # derive a fixed size embedding vector
            fixed_size = full_size[1:-1].mean(0).numpy()  # other than mean possible, too
            embeddings.append(fixed_size)
    embeddings = np.array(embeddings)

    return embeddings



if __name__=='__main__':

    # example dataset from the paper
    fasta_file = 'tests/head_10.fasta'

    # todo: re-write: this is super ugly -> parse as df directly
    headers = []
    sequences = []
    for h, s in read_fasta(fasta_file):
        headers.append(h)
        sequences.append(s)

    embeddings = gen_embedding(sequences, device='cuda')