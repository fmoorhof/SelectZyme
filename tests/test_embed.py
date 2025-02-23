from __future__ import annotations

import pytest

import embed


class TestSelectPLMModel:
    def test_select_esm1b_model(self):
        tokenizer, model = embed._select_plm_model("esm1b")
        assert tokenizer.name_or_path == "facebook/esm1b_t33_650M_UR50S"
        assert model.config._name_or_path == "facebook/esm1b_t33_650M_UR50S"

    def test_select_esm2_model(self):
        tokenizer, model = embed._select_plm_model("esm2")
        assert tokenizer.name_or_path == "facebook/esm2_t33_650M_UR50D"
        assert model.config._name_or_path == "facebook/esm2_t33_650M_UR50D"

    def test_select_prott5_model(self):
        tokenizer, model = embed._select_plm_model("prott5")
        assert tokenizer.name_or_path == "Rostlab/prot_t5_xl_uniref50"
        assert model.config._name_or_path == "Rostlab/prot_t5_xl_uniref50"

    def test_select_prostt5_model(self):
        tokenizer, model = embed._select_plm_model("prostt5")
        assert tokenizer.name_or_path == "Rostlab/ProstT5"
        assert model.config._name_or_path == "Rostlab/ProstT5"

    def test_select_invalid_model(self):
        with pytest.raises(
            ValueError,
            match="Model invalid_model not supported. Please choose one of: 'esm1b', 'esm2', 'prott5', 'prostt5'",
        ):
            embed._select_plm_model("invalid_model")


class TestGenEmbedding:
    """Test the gen_embedding function."""

    @pytest.mark.parametrize("no_pad", [True, False])
    def test_gen_embedding_esm1b(self, no_pad):
        sequences = ["MKTIIALSYIFCLVFADYKDDDDK", "MKAILVVLLYTFATANAD"]
        embeddings = embed.gen_embedding(sequences, plm_model="esm1b", no_pad=no_pad)
        assert embeddings.shape == (2, 1280)

    @pytest.mark.parametrize("no_pad", [True, False])
    def test_gen_embedding_esm2(self, no_pad):
        sequences = ["MKTIIALSYIFCLVFADYKDDDDK", "MKAILVVLLYTFATANAD"]
        embeddings = embed.gen_embedding(sequences, plm_model="esm2", no_pad=no_pad)
        assert embeddings.shape == (2, 1280)

    @pytest.mark.parametrize("no_pad", [True, False])
    def test_gen_embedding_prott5(self, no_pad):
        sequences = ["MKTIIALSYIFCLVFADYKDDDDK", "MKAILVVLLYTFATANAD"]
        embeddings = embed.gen_embedding(sequences, plm_model="prott5", no_pad=no_pad)
        assert embeddings.shape == (2, 1024)

    @pytest.mark.parametrize("no_pad", [True, False])
    def test_gen_embedding_prostt5(self, no_pad):
        sequences = ["MKTIIALSYIFCLVFADYKDDDDK", "MKAILVVLLYTFATANAD"]
        embeddings = embed.gen_embedding(sequences, plm_model="prostt5", no_pad=no_pad)
        assert embeddings.shape == (2, 1024)

    def test_gen_embedding_invalid_model(self):
        sequences = ["MKTIIALSYIFCLVFADYKDDDDK", "MKAILVVLLYTFATANAD"]
        with pytest.raises(
            ValueError,
            match="Model invalid_model not supported. Please choose one of: 'esm1b', 'esm2', 'prott5', 'prostt5'",
        ):
            embed.gen_embedding(sequences, plm_model="invalid_model")

    def test_gen_embedding_empty_sequences(self):
        sequences = []
        embeddings = embed.gen_embedding(sequences, plm_model="esm1b")
        assert embeddings.shape == (0,)

    def test_gen_embedding_single_sequence(self):
        sequences = ["MKTIIALSYIFCLVFADYKDDDDK"]
        embeddings = embed.gen_embedding(sequences, plm_model="esm1b")
        assert embeddings.shape == (1, 1280)
