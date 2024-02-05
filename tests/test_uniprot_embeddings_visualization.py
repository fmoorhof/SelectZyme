import logging
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use cuda:1, needs to be set before importing others
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # debugging GPU

import pandas as pd
import dash

import visualizer
from dash_app import run_dash_app
from load_uniprot_embeddings import create_db_from_5h


logging.basicConfig(
    format="%(levelname)-8s| %(module)s.%(funcName)s: %(message)s", level=logging.DEBUG
)

import h5py  # todo: not yet appended to requirements.txt
from tqdm import tqdm
import requests
import pandas as pd
import numpy as np


def fetch_uniprot_data(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.txt"
    response = requests.get(url)
    data = response.text.split('\n')

    length = None
    organism_id = None
    ec_number = None

    for line in data:
        if line.startswith('SQ'):
            length = int(line.split()[2])
        if line.startswith('OX'):
            organism_id = line.split('=')[1].split(';')[0]
        if line.startswith('DR') and 'EC' in line:
            ec_number = line.split(';')[2].strip()

    return length, organism_id, ec_number


def load_from_5h(filename: str) -> list:
    # Load the h5 data
    f = h5py.File(filename, 'r')

    entries = list(f.keys())
    # entries = entries[0:6]  # small test dataset for debugging (same size like head_10.tsv after preprocessing)
    vector_size = f[entries[0]].shape[0]
    logging.info(f"The vectors are of dimension: {vector_size}")
    logging.info(f"Got {len(entries)} entries from {filename}.")

    X = []
    annotations = []
    for i, entry in enumerate(tqdm(entries)):
        vector = f[entry][:].tolist()
        annotation = f.get(entry).attrs["original_id"]
        X.append(vector)
        annotations.append(annotation)
    X = np.array(X)

    return annotations, X


def main(annotations, X, project_name: str, app):

    df = pd.DataFrame(annotations, columns=['Entry'])

    sys.setrecursionlimit(max(df.shape[0], 10000))  # fixed: RecursionError: maximum recursion depth exceeded
    labels = visualizer.clustering_HDBSCAN(X, min_samples=50)  # 50
    df['cluster'] = labels  # add cluster labels to df (from clustering)
    # mock data for the plotly dash app
    df['EC number'] = '0.0.0.0'
    df['marker_size'] = 5
    df['marker_symbol'] = 'circle'

    # todo: annotations are garbage in the dataset, fetch own annoations from uniprot based on the 'Entry' column
    # df['Length'], df['Organism ID'], df['EC number'] = zip(*df['Entry'].map(fetch_uniprot_data))

    iter_methods = ['PCA', 'TSNE', 'UMAP']
    for method in iter_methods:
        if method == 'PCA':
            X_red = visualizer.pca(X)
        elif method == 'TSNE':
            X_red = visualizer.tsne(X)
        elif method == 'UMAP':
            X_red = visualizer.umap(X)
        visualizer.plot_2d(df, X_red, collection_name=project_name, method=method)


    # app = run_dash_app(df, X_red, method, project_name, app)



if __name__ == "__main__":
    # todo: fix that the app is constantly re-starting itself and calling the main() function again:/
    app = dash.Dash(__name__)
    # main(input_file='tests/head_10.tsv', project_name='test_project', app=app)

    from load_uniprot_embeddings import create_db_from_5h
    # annotations, X = create_db_from_5h('/scratch/global_1/fmoorhof/Databases/per-protein.h5', 'swiss-prot2024-01-14')
    annotations, X = load_from_5h('/scratch/global_1/fmoorhof/Databases/per-protein.h5')
    main(annotations, X, project_name='swiss-prot2024-01-14', app=app)  # test uniprot embeddings

    # app.run_server(host='0.0.0.0', port=8051, debug=True)  # from docker (no matter is docker or not) to local machine: http://192.168.3.156:8051/