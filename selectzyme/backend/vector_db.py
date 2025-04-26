from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from selectzyme.backend.embed import gen_embedding
from selectzyme.backend.utils import run_time


class QdrantDB:
    """A class to manage Qdrant vector database operations."""
    def __init__(self, collection_name: str, host: str = "http://localhost:6333"):
        """
        Initialize the Qdrant client and collection settings.
        
        :param collection_name: Name of the collection in Qdrant
        :param host: URL of the Qdrant server
        """
        self.collection_name = collection_name
        self.qdrant = QdrantClient(url=host, prefer_grpc=True)

    def collection_exists(self) -> bool:
        """Check if the collection already exists in Qdrant."""
        collections_info = self.qdrant.get_collections()
        collection_names = {col.name for col in collections_info.collections}
        return self.collection_name in collection_names

    @run_time
    def database_access(self, df: pd.DataFrame, plm_model: str = "esm1b") -> np.ndarray:
        """
        Create or retrieve a collection in Qdrant DB with embedded sequences.
        
        :param df: DataFrame containing sequences.
        :param plm_model: Model used for generating embeddings.
        :return: Numpy array of embeddings.
        """
        logging.info("Connecting to Qdrant vector database...")

        if not self.collection_exists():  # If collection does not exist, create it
            logging.info(f"Creating new collection: {self.collection_name}")
            embeddings = gen_embedding(df["sequence"].tolist(), plm_model=plm_model)
            self.create_collection(embeddings)
            self.upload_points(embeddings)

        else:  # Load existing collection
            logging.info(f"Collection {self.collection_name} found. Loading embeddings...")
            embeddings = self.load_collection()

            if df.shape[0] != embeddings.shape[0]:
                logging.warning(
                    f"Mismatch in DataFrame ({df.shape[0]}) and embeddings ({embeddings.shape[0]})."
                    " Replacing collection with new data."
                )
                self.qdrant.delete_collection(self.collection_name)
                embeddings = gen_embedding(df["sequence"].tolist(), plm_model=plm_model)
                self.create_collection(embeddings)
                self.upload_points(embeddings)

        sys.setrecursionlimit(
            max(df.shape[0], 10000)
        )  # fixed: RecursionError: maximum recursion depth exceeded
        
        return embeddings

    def create_collection(self, embeddings: np.ndarray) -> None:
        """
        Create a new collection in Qdrant with the given embeddings.
        
        :param embeddings: Numpy array of embeddings.
        """
        logging.info(f"Creating Qdrant collection: {self.collection_name}")
        self.qdrant.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=embeddings.shape[1],  # Vector size is defined by used model
                distance=models.Distance.EUCLID,
            ),
        )

    def upload_points(self, embeddings: np.ndarray) -> None:
        """
        Upload embedding points to the Qdrant collection.
        
        :param embeddings: Numpy array of embeddings.
        """
        logging.info(f"Uploading {len(embeddings)} embeddings to {self.collection_name}...")

        records = [
            models.Record(id=i, vector=embedding.tolist()) for i, embedding in enumerate(embeddings)
        ]
        self.qdrant.upload_records(collection_name=self.collection_name, records=records)

    def load_collection(self) -> np.ndarray:
        """
        Load all embeddings from the Qdrant collection.
        
        :return: Numpy array of embeddings.
        """
        logging.info(f"Loading collection: {self.collection_name}")
        collection = self.qdrant.get_collection(self.collection_name)
        records, _ = self.qdrant.scroll(
            collection_name=self.collection_name,
            with_vectors=True,
            limit=collection.points_count,
            timeout=190,
        )

        embeddings = np.array([record.vector for record in tqdm(records)])
        return embeddings
