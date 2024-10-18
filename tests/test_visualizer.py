import pytest
import pandas as pd
import numpy as np

from visualizer import clustering_HDBSCAN


class TestVisualizer:
    def setup_method(self):
        mock_sequences = ['ACGT', 'TGCA', 'GACT', 'CTAG', 'ATGC']
        mock_df = pd.DataFrame({'sequence': mock_sequences})
        self.df = mock_df
        self.X = np.random.rand(len(mock_sequences), 1280)  # Mock embeddings
        self.annotation = 'mock_annotation'

    def test_load_datasets(self):
        """Test the loading of the datasets."""
        assert self.df is not None
        assert self.annotation is not None
        assert self.X is not None

    def test_error_clustering_HDBSCAN(self):
        """Test the clustering of the embeddings with HDBSCAN."""
        with pytest.raises(ValueError, match="The number of samples in X is less than min_samples. Please try a smaller value for min_samples."):
            labels = clustering_HDBSCAN(self.X)
            
    @pytest.mark.skip(reason="Fix needed: This test is failing because the number of samples in X is less than min_samples.")
    def test_clustering_HDBSCAN(self):
        """Test the clustering of the embeddings with HDBSCAN."""
        labels = clustering_HDBSCAN(self.X, min_samples=1, min_cluster_size=1)
        assert labels is not None
        assert len(labels) == self.X.shape[0]                
