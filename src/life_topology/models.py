from __future__ import annotations

from sentence_transformers import SentenceTransformer

from .config import MODEL_REVISIONS, SAFETENSORS_MODELS


def load_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Load only audited, immutable model artifacts with remote code disabled."""
    try:
        revision = MODEL_REVISIONS[model_name]
    except KeyError as exc:
        raise ValueError(f"Embedding model is not in the audited allowlist: {model_name}") from exc

    kwargs: dict[str, object] = {
        "revision": revision,
        "trust_remote_code": False,
    }
    if model_name in SAFETENSORS_MODELS:
        kwargs["model_kwargs"] = {"use_safetensors": True}
    return SentenceTransformer(model_name, **kwargs)
