"""Build / load FAISS vector store on disk."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .paths import VECTOR_STORE_DIR, ensure_dirs

logger = logging.getLogger(__name__)


def vector_store_path(repo_id: int) -> Path:
    ensure_dirs()
    return VECTOR_STORE_DIR / str(repo_id)


def build_and_save(
    *,
    repo_id: int,
    documents: Sequence[Document],
    ids: Sequence[str],
    embeddings: Embeddings,
) -> Path:
    """Embed `documents`, build a FAISS index and persist it.

    Returns the directory where the index was saved.
    """
    if len(documents) != len(ids):
        raise ValueError("documents 和 ids 长度必须一致")
    if not documents:
        raise ValueError("没有可索引的 chunk")

    path = vector_store_path(repo_id)
    logger.info("构建 FAISS 索引 (chunk 数=%d)", len(documents))
    store = FAISS.from_documents(documents=list(documents), embedding=embeddings, ids=list(ids))
    path.mkdir(parents=True, exist_ok=True)
    logger.info("保存 FAISS 索引到 %s", path)
    store.save_local(str(path))
    return path
