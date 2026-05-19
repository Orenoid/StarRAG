"""Embedding factory.

Prefer a local HuggingFace sentence-transformer model so we don't need an
API key. If loading the local model fails (e.g. no internet on first run
and the model isn't cached), fall back to OpenAI when `OPENAI_API_KEY` is
present.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


LOCAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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
            logger.info("加载本地嵌入模型: %s", LOCAL_MODEL_NAME)
            emb = HuggingFaceEmbeddings(
                model_name=LOCAL_MODEL_NAME,
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
