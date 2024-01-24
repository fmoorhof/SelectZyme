import pytest
import pandas as pd
from qdrant_client import QdrantClient

import visualizer
from embed import load_collection_from_vector_db


class TestVisualizer:
    def setup_method(self):
        self.df = pd.read_csv('tests/head_10.tsv')
        qdrant = QdrantClient(path="datasets/Vector_db/")  # OR write them to disk
        self.annotation, self.X = load_collection_from_vector_db(qdrant, collection_name='pytest')  # X = embeddings

    def test_error_clustering_HDBSCAN(self):
        """Test the clustering of the embeddings with HDBSCAN."""
        with pytest.raises(ValueError, match="The number of samples in X is less than min_samples. Please try a smaller value for min_samples."):
            labels = visualizer.clustering_HDBSCAN(self.X)
            
    def test_clustering_HDBSCAN(self):
        """Test the clustering of the embeddings with HDBSCAN."""
        labels = visualizer.clustering_HDBSCAN(self.X, min_samples=1)
        assert labels is not None
        assert len(labels) == self.X.shape[0]                
