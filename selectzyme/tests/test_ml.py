from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backend.ml import dimred_caller, perform_hdbscan_clustering


class TestML(unittest.TestCase):
    def setUp(self):
        # Setup dummy data for tests
        self.X = np.random.rand(100, 10)  # Example data
        self.X_centroids = np.random.rand(5, 10)  # Example centroids
        self.df = pd.DataFrame(self.X, columns=[f"feature_{i}" for i in range(self.X.shape[1])])

    def test_perform_hdbscan_clustering(self):
        # Act
        mst, linkage, df = perform_hdbscan_clustering(self.X, self.df)

        # Assert
        self.assertIsInstance(mst, np.ndarray)
        self.assertIsInstance(linkage, np.ndarray)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("cluster", df.columns)
        self.assertIn("marker_symbol", df.columns)

        # Test error handling
        with self.assertRaises(ValueError):
            perform_hdbscan_clustering(self.X[:10], self.df[:10], min_samples=20)

    def test_dimred_caller_pca(self):
        # Act
        X_red = dimred_caller(self.X, dim_method="PCA")

        # Assert
        self.assertIsInstance(X_red, np.ndarray)
        self.assertEqual(X_red.shape[1], 2)

    def test_dimred_caller_tsne(self):
        # Act
        X_red = dimred_caller(self.X, dim_method="TSNE")

        # Assert
        self.assertIsInstance(X_red, np.ndarray)
        self.assertEqual(X_red.shape[1], 2)

    def test_dimred_caller_opentsne(self):
        # Act
        X_red = dimred_caller(self.X, dim_method="OPENTSNE")

        # Assert
        self.assertIsInstance(X_red, np.ndarray)
        self.assertEqual(X_red.shape[1], 2)

    def test_dimred_caller_umap(self):
        # Act
        X_red = dimred_caller(self.X, dim_method="UMAP")

        # Assert
        self.assertIsInstance(X_red, np.ndarray)
        self.assertEqual(X_red.shape[1], 2)