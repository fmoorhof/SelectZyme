from __future__ import annotations

import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.mst_plotting import MinimumSpanningTree


class TestMinimumSpanningTree(unittest.TestCase):
    def setUp(self):
        self.sample_size = 5
        self.df = pd.DataFrame(
            {
                "x": np.random.randn(self.sample_size),
                "y": np.random.randn(self.sample_size),
                "accession": np.random.choice(["A", "B", "C"], self.sample_size),
                "cluster": np.random.choice(["A", "B", "C"], self.sample_size),
                "selected": np.random.choice([False, True], self.sample_size),
                "marker_symbol": np.random.choice(
                    ["circle", "square", "diamond", "triangle-up"], self.sample_size
                ),
                "marker_size": np.random.randint(2, self.sample_size, self.sample_size),
            }
        )
        self.X_red = np.random.randn(self.sample_size, 2)
        self.mst = np.array([[0, 1, 0.5], [1, 2, 0.3], [2, 3, 0.2], [3, 4, 0.1]])
        self.fig = go.Figure()
        self.mst_plotter = MinimumSpanningTree(self.mst, self.df, self.X_red, self.fig)

    def test_plot_mst_force_directed(self):
        G = nx.Graph()
        G.add_weighted_edges_from(self.mst)
        fig = self.mst_plotter.plot_mst_force_directed(G)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)  # edge_trace and node_trace

    def test_plot_mst_in_dimred_landscape(self):
        fig = self.mst_plotter.plot_mst_in_dimred_landscape()
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)  # edge_trace

    def test_create_edge_trace(self):
        edge_x = [0, 1, None, 1, 2, None]
        edge_y = [0, 1, None, 1, 2, None]
        edge_trace = self.mst_plotter.create_edge_trace(edge_x, edge_y)
        self.assertIsInstance(edge_trace, go.Scattergl)
        self.assertEqual(edge_trace.mode, "lines")

    def test_create_node_trace(self):
        node_x = np.random.randn(self.sample_size)
        node_y = np.random.randn(self.sample_size)
        node_adjacencies = np.random.randint(1, 5, self.sample_size)
        node_trace = self.mst_plotter.create_node_trace(
            node_x, node_y, node_adjacencies
        )
        self.assertIsInstance(node_trace, go.Scattergl)
        self.assertEqual(node_trace.mode, "markers")

    @patch("src.mst_plotting.set_columns_of_interest")
    def test_modify_graph_data(self, mock_set_columns_of_interest):
        mock_set_columns_of_interest.return_value = ["accession", "cluster"]
        G = nx.Graph()
        G.add_weighted_edges_from(self.mst)
        edge_trace, node_trace = self.mst_plotter._modify_graph_data(G)
        self.assertIsInstance(edge_trace, go.Scattergl)
        self.assertIsInstance(node_trace, go.Scattergl)
        self.assertEqual(len(node_trace.marker.color), self.sample_size)


if __name__ == "__main__":
    unittest.main()
