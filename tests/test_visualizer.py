from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.visualizer import plot_2d


class TestVisualizer(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "accession": ["P12345", "P67890"],
                "species": ["Human", "Mouse"],
                "marker_size": [10, 20],
                "marker_symbol": ["circle", "square"],
                "legend_attribute": ["A", "B"],
            }
        )
        self.X_red = np.array([[1, 2], [3, 4]])

    def test_plot_2d(self):
        fig = plot_2d(self.df, self.X_red, "legend_attribute")
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(
            len(fig.data), 6
        )  # Two traces for data point categories, four for marker symbols defined in visualizer.py

        # Check the first trace
        trace = fig.data[0]
        self.assertEqual(trace.name, "A")
        self.assertEqual(trace.x.tolist(), [1])
        self.assertEqual(trace.y.tolist(), [2])
        self.assertEqual(trace.marker.size.tolist(), [10])
        self.assertEqual(trace.marker.symbol, "circle")

        # Check the second trace
        trace = fig.data[1]
        self.assertEqual(trace.name, "B")
        self.assertEqual(trace.x.tolist(), [3])
        self.assertEqual(trace.y.tolist(), [4])
        self.assertEqual(trace.marker.size.tolist(), [20])
        self.assertEqual(trace.marker.symbol, "square")


if __name__ == "__main__":
    unittest.main()
