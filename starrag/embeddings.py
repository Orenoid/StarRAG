"""Embedding factory.

Prefer a local HuggingFace sentence-transformer model so we don't need an
API key. If loading the local model fails (e.g. no internet on first run
and the model isn't cached), fall back to OpenAI when `OPENAI_API_KEY` is
present.

Caches:
- `MODEL_CACHE_DIR` (project-local) keeps the downloaded sentence-transformer
  weights on disk so they survive across CLI invocations.
- `@lru_cache` ensures the same Python process only instantiates the model
  once in memory.
"""
from __future__ import annotations

import functools
import logging
import os

from .paths import MODEL_CACHE_DIR, ensure_dirs

logger = logging.getLogger(__name__)


LOCAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _is_model_cached() -> bool:
    """Return True if the model has already been downloaded to MODEL_CACHE_DIR."""
    marker = MODEL_CACHE_DIR / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots"
    return marker.exists() and any(marker.iterdir())


@functools.lru_cache(maxsize=1)
def build_embeddings():
    """Return a LangChain Embeddings instance.

    Tries local HuggingFace first, falls back to OpenAI on failure.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        logger.warning("langchain_huggingface 未安装: %s", e)
    else:
        try:
            ensure_dirs()
            logger.info("加载本地嵌入模型: %s", LOCAL_MODEL_NAME)
            local_only = _is_model_cached()
            if local_only:
                logger.info("模型已缓存,跳过 HuggingFace Hub 检查")
            emb = HuggingFaceEmbeddings(
                model_name=LOCAL_MODEL_NAME,
                cache_folder=str(MODEL_CACHE_DIR),
                model_kwargs={"local_files_only": local_only},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("本地嵌入模型就绪")
            return emb
        except Exception as e:
            logger.warning("加载本地嵌入模型失败,尝试 OpenAI 回退: %s", e)

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "本地嵌入模型不可用,且未设置 OPENAI_API_KEY 用作回退。"
            "请安装 sentence-transformers 或设置 OPENAI_API_KEY。"
        )
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as e:
        raise RuntimeError(
            "OpenAI 回退失败:未安装 langchain-openai。"
            "请运行 `uv add langchain-openai`。"
        ) from e
    logger.info("使用 OpenAI text-embedding-3-small 作为回退")
    return OpenAIEmbeddings(model="text-embedding-3-small")
