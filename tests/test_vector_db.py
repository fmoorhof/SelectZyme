import pytest
from qdrant_client import QdrantClient

import vector_db
from preprocessing import Parsing
from preprocessing import Preprocessing


@pytest.fixture(scope="class")
def setup_and_teardown():
    """Fixture for setting up and tearing down resources."""
    # Setup: This code runs before the tests in the class.
    print("Setting up for the test...")
    qdrant = QdrantClient(path="datasets/Vector_db/")
    collection_name = 'pytest'

    yield qdrant, collection_name  # The test functions will run at this point.

    # Teardown: This code runs after the tests in the class.
    print("Tearing down after the test...")
    # qdrant.delete_collection(collection_name='pytest')
    qdrant.close()


@pytest.mark.usefixtures("setup_and_teardown")
class TestDBCreation:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        # Replace this with the code to create your DataFrame
        self.df = Parsing.parse_tsv('tests/head_10.tsv')
        pp = Preprocessing(self.df)
        pp.remove_long_sequences()
        self.df = pp.df
        self.embeddings = vector_db.gen_embedding(self.df['sequence'].tolist())

    def test_create_vector_db_collection(self, setup_and_teardown):
        """Test the creation of a vector database collection."""
        qdrant, collection_name = setup_and_teardown
        vector_db.create_vector_db_collection(qdrant, self.df, self.embeddings, collection_name)
        collections_info = qdrant.get_collections()

        assert collection_name in str(collections_info)
        assert collection_name == collections_info.collections[0].name

    def test_load_collection_from_vector_db(self, setup_and_teardown):
        """Test the loading of an existing vector database collection."""
        qdrant, collection_name = setup_and_teardown
        entries, embeddings = vector_db.load_collection_from_vector_db(qdrant, collection_name)

        assert entries is not None
        assert embeddings is not None
        assert len(entries) == self.df.shape[0]

    def test_collection_deletion(self, setup_and_teardown):
        """Test the deletion of a collection."""
        qdrant, collection_name = setup_and_teardown
        res = qdrant.delete_collection(collection_name=collection_name)
        collections_info = qdrant.get_collections()

        assert res is True
        assert collection_name not in str(collections_info)
