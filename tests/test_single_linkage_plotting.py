import pytest
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
import plotly.graph_objects as go

from src.single_linkage_plotting import create_dendrogram, _value_to_color
from src.customizations import set_columns_of_interest


@pytest.fixture
def sample_data():
    np.random.seed(42)
    sample_size = 11
    df = pd.DataFrame({
        'x': np.random.randn(sample_size),
        'y': np.random.randn(sample_size),
        'accession': np.arange(sample_size),
        'cluster': np.random.choice([-1, 1, 3], sample_size),
        'selected': np.random.choice([False, True], sample_size),
        'marker_symbol': np.random.choice(['circle', 'square', 'diamond', 'triangle-up'], sample_size),
        'marker_size': np.random.randint(10, sample_size, sample_size)
    })
    data = np.random.randn(sample_size, 2)
    Z = linkage(data, method='single')
    return Z, df

def test_create_dendrogram(sample_data):
    Z, df = sample_data
    fig = create_dendrogram(Z, df)
    
    assert fig is not None
    assert isinstance(fig, go.Figure)
    assert 'layout' in fig.to_plotly_json()
    assert 'data' in fig.to_plotly_json()

def test_value_to_color():
    values = np.array([1, 2, 3, 4, 5])
    color_mapping = _value_to_color(values)
    
    assert isinstance(color_mapping, dict)
    assert len(color_mapping) == len(np.unique(values))
    for value in values:
        assert value in color_mapping
        assert isinstance(color_mapping[value], str)  # Check if the color is a string

def test_value_to_color_single_value():
    values = np.array([1, 1, 1])
    color_mapping = _value_to_color(values)
    
    assert isinstance(color_mapping, dict)
    assert len(color_mapping) == 1
    assert 1 in color_mapping
    assert isinstance(color_mapping[1], str)  # Check if the color is a string