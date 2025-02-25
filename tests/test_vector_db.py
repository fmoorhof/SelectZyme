from __future__ import annotations

import pytest
from qdrant_client import QdrantClient
import numpy as np

import vector_db
from parsing import Parsing
from preprocessing import Preprocessing


@pytest.fixture(scope="class")
def setup_and_teardown():
    """Fixture for setting up and tearing down resources."""
    # Setup: This code runs before the tests in the class.
    print("Setting up for the test...")
    qdrant = QdrantClient(location=":memory:")
    collection_name = "_pytest"

    yield qdrant, collection_name  # The test functions will run at this point.

    # Teardown: This code runs after the tests in the class.
    print("Tearing down after the test...")
    qdrant.delete_collection(collection_name='_pytest')
    qdrant.close()


@pytest.mark.usefixtures("setup_and_teardown")
class TestDBCreation:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        # Replace this with the code to create your DataFrame
        self.df = Parsing("tests/head_10.tsv").parse_tsv()
        pp = Preprocessing(self.df)
        pp.remove_long_sequences()
        self.df = pp.df
        self.embeddings = vector_db.gen_embedding(self.df["sequence"].tolist())

    def test_database_access(self, setup_and_teardown):
        """Test the database access function."""
        qdrant, collection_name = setup_and_teardown
        embeddings = vector_db.database_access(self.df, collection_name)

        assert embeddings is not None
        assert len(embeddings) == self.df.shape[0]

    def test_create_collection(self, setup_and_teardown):
        """Test the creation of a collection in Qdrant."""
        qdrant, collection_name = setup_and_teardown
        embeddings = np.random.rand(10, 128)  # Example embeddings
        vector_db.create_collection(qdrant, embeddings, collection_name)
        collections_info = qdrant.get_collections()

        assert collection_name in [collection.name for collection in collections_info.collections]

    def test_upload_points(self, setup_and_teardown):
        """Test uploading points to a collection in Qdrant."""
        qdrant, collection_name = setup_and_teardown
        embeddings = np.random.rand(10, 128)  # Example embeddings
        vector_db.create_collection(qdrant, embeddings, collection_name)
        vector_db.upload_points(qdrant, embeddings, collection_name)
        collection = qdrant.get_collection(collection_name)
        
        assert collection.points_count == len(embeddings)

    def test_load_collection_from_vector_db(self, setup_and_teardown):
        """Test loading a collection from Qdrant."""
        qdrant, collection_name = setup_and_teardown
        embeddings = np.random.rand(10, 128)  # Example embeddings
        vector_db.create_collection(qdrant, embeddings, collection_name)
        vector_db.upload_points(qdrant, embeddings, collection_name)
        loaded_embeddings = vector_db.load_collection_from_vector_db(qdrant, collection_name)
        
        # different precision lets assert equal fail, todo: fix that somehow
        # assert np.array_equal(embeddings, loaded_embeddings)
        assert np.allclose(embeddings, loaded_embeddings)

    def test_collection_deletion(self, setup_and_teardown):
        """Test the deletion of a collection."""
        qdrant, collection_name = setup_and_teardown
        res = qdrant.delete_collection(collection_name=collection_name)
        collections_info = qdrant.get_collections()

        assert res is True
        assert collection_name not in str(collections_info)        