from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qdrant_client import QdrantClient

import vector_db


@pytest.fixture(scope="class")
def setup_and_teardown():
    """Fixture for setting up and tearing down resources."""
    # Setup: This code runs before the tests in the class.
    print("Setting up for the test...")
    qdrant = QdrantClient()
    collection_name = "_pytest"

    yield qdrant, collection_name  # The test functions will run at this point.

    # Teardown: This code runs after the tests in the class.
    print("Tearing down after the test...")
    qdrant.delete_collection(collection_name='_pytest')
    qdrant.close()


@pytest.mark.usefixtures("setup_and_teardown")
class TestQdrantDB:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.df = pd.DataFrame({
                "accession": ["seq1", "seq2"],
                "sequence": ["MAEAEMMAEA", "MAEAE"]
            })

        self.embeddings = vector_db.gen_embedding(self.df["sequence"].tolist())
        self.collection_name = "_pytest"
        self.qdrant_db = vector_db.QdrantDB(collection_name=self.collection_name)

    def test_init(self):
        """Test the initialization of QdrantDB."""
        assert self.qdrant_db.collection_name == self.collection_name
        assert isinstance(self.qdrant_db.qdrant, QdrantClient)

    def test_database_access(self, setup_and_teardown):
        """Test the database access function."""
        qdrant, collection_name = setup_and_teardown
        embeddings = self.qdrant_db.database_access(self.df)

        assert embeddings is not None
        assert len(embeddings) == self.df.shape[0]

    def test_create_collection(self, setup_and_teardown):
        """Test the creation of a collection in Qdrant."""
        qdrant, collection_name = setup_and_teardown
        embeddings = np.random.rand(10, 128)  # Example embeddings
        self.qdrant_db.create_collection(embeddings)
        collections_info = qdrant.get_collections()

        assert collection_name in [collection.name for collection in collections_info.collections]

    def test_upload_points(self, setup_and_teardown):
        """Test uploading points to a collection in Qdrant."""
        qdrant, collection_name = setup_and_teardown
        embeddings = np.random.rand(10, 128)  # Example embeddings
        self.qdrant_db.create_collection(embeddings)
        self.qdrant_db.upload_points(embeddings)
        collection = qdrant.get_collection(collection_name)
        
        assert collection.points_count == len(embeddings)

    def test_load_collection(self, setup_and_teardown):
        """Test loading a collection from Qdrant."""
        qdrant, collection_name = setup_and_teardown
        embeddings = np.random.rand(10, 128)  # Example embeddings
        self.qdrant_db.create_collection(embeddings)
        self.qdrant_db.upload_points(embeddings)
        loaded_embeddings = self.qdrant_db.load_collection()
        
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