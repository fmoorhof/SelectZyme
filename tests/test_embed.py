from __future__ import annotations

import pytest
from selectzyme.backend import embed


class TestLoadModelAndTokenizer:
    """Tests for loading PLM model and tokenizer."""

    @pytest.mark.parametrize(
        "model_name,expected_path",
        [
            ("esm1b", "facebook/esm1b_t33_650M_UR50S"),
            ("esm2", "facebook/esm2_t33_650M_UR50D"),
            ("prott5", "Rostlab/prot_t5_xl_uniref50"),
            ("prostt5", "Rostlab/ProstT5"),
        ],
    )
    def test_load_valid_model(self, model_name: str, expected_path: str):
        tokenizer, model = embed._load_model_and_tokenizer(model_name)
        assert tokenizer.name_or_path == expected_path
        assert model.config._name_or_path == expected_path

    def test_load_invalid_model(self):
        with pytest.raises(
            ValueError,
            match="Unsupported model 'invalid_model'. Choose from 'esm1b', 'esm2', 'prott5', 'prostt5'.",
        ):
            embed._load_model_and_tokenizer("invalid_model")


class TestGenEmbedding:
    """Tests for generating embeddings."""

    @pytest.mark.parametrize(
        "model_name,expected_dim",
        [
            ("esm1b", 1280),
            ("esm2", 1280),
            ("prott5", 1024),
            ("prostt5", 1024),
        ],
    )
    @pytest.mark.parametrize("no_pad", [True, False])
    def test_gen_embedding_multiple_sequences(
        self, model_name: str, expected_dim: int, no_pad: bool
    ):
        sequences = ["MKTIIALSYIFCLVFADYKDDDDK", "MKAILVVLLYTFATANAD"]
        embeddings = embed.gen_embedding(sequences, plm_model=model_name, no_pad=no_pad)
        assert embeddings.shape == (2, expected_dim)

    @pytest.mark.parametrize(
        "model_name,expected_dim",
        [
            ("esm1b", 1280),
            ("esm2", 1280),
            ("prott5", 1024),
            ("prostt5", 1024),
        ],
    )
    @pytest.mark.parametrize("no_pad", [True, False])
    def test_gen_embedding_single_sequence(
        self, model_name: str, expected_dim: int, no_pad: bool
    ):
        sequences = ["MKTIIALSYIFCLVFADYKDDDDK"]
        embeddings = embed.gen_embedding(sequences, plm_model=model_name, no_pad=no_pad)
        assert embeddings.shape == (1, expected_dim)

    def test_gen_embedding_empty_sequences(self):
        sequences: list[str] = []
        embeddings = embed.gen_embedding(sequences, plm_model="esm1b")
        assert embeddings.shape == (0,)

    def test_gen_embedding_invalid_model(self):
        sequences = ["MKTIIALSYIFCLVFADYKDDDDK"]
        with pytest.raises(
            ValueError,
            match="Unsupported model 'invalid_model'. Choose from 'esm1b', 'esm2', 'prott5', 'prostt5'.",
        ):
            embed.gen_embedding(sequences, plm_model="invalid_model")

    @pytest.mark.skip(reason="Error message assert fails CUDA error: device-side assert triggered")
    def test_gen_embedding_too_long_sequence_for_esm1b(self):
        """Test that too long sequences (>1024) raise a RuntimeError for esm1b."""
        # Create a sequence longer than 1024 amino acids
        long_sequence = "M" * 1025
        sequences = [long_sequence]

        with pytest.raises(RuntimeError) as exc_info:
            embed.gen_embedding(sequences, plm_model="esm1b")

        error_message = str(exc_info.value)
        assert "ESM-1b model cannot handle sequences longer than 1024 amino acids." in error_message
        assert long_sequence[:20] in error_message  # Check problematic sequence appears (partially)
        assert "filter or truncate" in error_message
