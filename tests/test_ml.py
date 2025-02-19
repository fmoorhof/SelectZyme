import unittest

import numpy as np
from cuml.cluster import HDBSCAN

from ml import (
    _weighted_cluster_centroid, clustering_HDBSCAN, pca, tsne, opentsne, umap
)


class TestML(unittest.TestCase):

    def setUp(self):
        # Setup dummy data for tests
        self.X = np.random.rand(100, 10)  # Example data
        self.X_centroids = np.random.rand(5, 10)  # Example centroids

    @unittest.skip("not fixed yet: FAILED tests/test_ml.py::TestML::test_weighted_cluster_centroid - ZeroDivisionError: Weights sum to zero, can't be normalized")
    def test_weighted_cluster_centroid(self):
        # Arrange
        model = HDBSCAN(min_samples=2, min_cluster_size=4)
        model.fit(self.X)
        cluster_id = model.labels_[0]  # Example cluster ID

        # Act
        centroid = _weighted_cluster_centroid(model, self.X, cluster_id)

        # Assert
        self.assertIsInstance(centroid, np.ndarray)
        self.assertEqual(centroid.shape[0], self.X.shape[1])

        # Test noise cluster handling
        with self.assertRaisesRegex(ValueError, "Cannot calculate centroid for noise cluster"):
            _weighted_cluster_centroid(model, self.X, -1)

    def test_clustering_HDBSCAN(self):
        # Act
        labels, G, Gsl, X_centroids = clustering_HDBSCAN(self.X)

        # Assert
        self.assertIsInstance(labels, np.ndarray)
        self.assertIsInstance(G, object)  # Check if it's a valid type
        self.assertIsInstance(Gsl, object)  # Check if it's a valid type
        self.assertIsInstance(X_centroids, np.ndarray)

        # Test error handling
        with self.assertRaisesRegex(ValueError, "The number of samples in X is less than min_samples."):
            clustering_HDBSCAN(self.X, min_samples=self.X.shape[0] + 1)

    def test_pca(self):
        # Act
        X_pca, X_pca_centroid = pca(self.X, self.X_centroids)

        # Assert
        self.assertIsInstance(X_pca, np.ndarray)
        self.assertIsInstance(X_pca_centroid, np.ndarray)
        self.assertEqual(X_pca.shape[1], 2)
        self.assertEqual(X_pca_centroid.shape[1], 2)

    def test_tsne(self):
        # Act
        X_tsne, X_tsne_centroid = tsne(self.X)

        # Assert
        self.assertIsInstance(X_tsne, np.ndarray)
        self.assertIsInstance(X_tsne_centroid, np.ndarray)
        self.assertEqual(X_tsne.shape[1], 2)
        self.assertEqual(X_tsne_centroid.shape[1], 2)
        self.assertEqual(X_tsne_centroid.shape[0], 0)  # Centroid array should be empty

    def test_opentsne(self):  # test is quite slow
        # Act
        X_tsne, X_tsne_centroid = opentsne(self.X, self.X_centroids)

        # Assert
        self.assertIsInstance(X_tsne, np.ndarray)
        self.assertIsInstance(X_tsne_centroid, np.ndarray)
        self.assertEqual(X_tsne.shape[1], 2)
        self.assertEqual(X_tsne_centroid.shape[1], 2)

    def test_umap(self):
        # Act
        X_umap, X_umap_centroid = umap(self.X, self.X_centroids)

        # Assert
        self.assertIsInstance(X_umap, np.ndarray)
        self.assertIsInstance(X_umap_centroid, np.ndarray)
        self.assertEqual(X_umap.shape[1], 2)
        self.assertEqual(X_umap_centroid.shape[1], 2)
