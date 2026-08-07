from __future__ import annotations

import unittest
from unittest.mock import patch

from life_topology.config import MODEL_REVISIONS
from life_topology.models import load_sentence_transformer


class ModelSecurityTests(unittest.TestCase):
    def test_minilm_is_revision_pinned_and_safetensors_only(self) -> None:
        name = "sentence-transformers/all-MiniLM-L6-v2"
        with patch("life_topology.models.SentenceTransformer") as constructor:
            load_sentence_transformer(name)
        constructor.assert_called_once_with(
            name,
            revision=MODEL_REVISIONS[name],
            trust_remote_code=False,
            model_kwargs={"use_safetensors": True},
        )

    def test_bge_is_revision_pinned_and_remote_code_disabled(self) -> None:
        name = "BAAI/bge-m3"
        with patch("life_topology.models.SentenceTransformer") as constructor:
            load_sentence_transformer(name)
        constructor.assert_called_once_with(
            name,
            revision=MODEL_REVISIONS[name],
            trust_remote_code=False,
        )

    def test_unreviewed_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "audited allowlist"):
            load_sentence_transformer("attacker/arbitrary-model")


if __name__ == "__main__":
    unittest.main()
