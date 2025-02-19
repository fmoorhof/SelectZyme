import unittest
import pandas as pd
import numpy as np
from src.visualizer import plot_2d

import plotly.graph_objects as go

class TestVisualizer(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'accession': ['P12345', 'P67890'],
            'species': ['Human', 'Mouse'],
            'marker_size': [10, 20],
            'marker_symbol': ['circle', 'square'],
            'legend_attribute': ['A', 'B']
        })
        self.X_red = np.array([[1, 2], [3, 4]])
        self.X_red_centroids = np.array([[2, 3]])

    def test_plot_2d(self):
        fig = plot_2d(self.df, self.X_red, self.X_red_centroids, 'legend_attribute')
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 3)  # Two traces for data points and one for centroids

        # Check the first trace
        trace = fig.data[0]
        self.assertEqual(trace.name, 'A')
        self.assertEqual(trace.x.tolist(), [1])
        self.assertEqual(trace.y.tolist(), [2])
        self.assertEqual(trace.marker.size.tolist(), [10])
        self.assertEqual(trace.marker.symbol, 'circle')

        # Check the second trace
        trace = fig.data[1]
        self.assertEqual(trace.name, 'B')
        self.assertEqual(trace.x.tolist(), [3])
        self.assertEqual(trace.y.tolist(), [4])
        self.assertEqual(trace.marker.size.tolist(), [20])
        self.assertEqual(trace.marker.symbol, 'square')

        # Check the centroid trace
        trace = fig.data[2]
        self.assertEqual(trace.name, 'Cluster Centroids')
        self.assertEqual(trace.x.tolist(), [2])
        self.assertEqual(trace.y.tolist(), [3])
        self.assertEqual(trace.marker.size, 10)
        self.assertEqual(trace.marker.symbol, 'x')
        self.assertEqual(trace.marker.color, 'red')

if __name__ == '__main__':
    unittest.main()