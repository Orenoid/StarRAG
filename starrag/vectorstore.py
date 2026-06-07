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


GLOBAL_VECTOR_STORE_NAME = "global"


def vector_store_path(repo_id: int | None = None) -> Path:
    """Return the single FAISS store shared by all repo chunks."""
    ensure_dirs()
    return VECTOR_STORE_DIR / GLOBAL_VECTOR_STORE_NAME


def _has_saved_store(path: Path) -> bool:
    return (path / "index.faiss").exists() and (path / "index.pkl").exists()


def _load_store(path: Path, embeddings: Embeddings) -> FAISS:
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def delete_ids(*, ids: Sequence[str], embeddings: Embeddings) -> None:
    """Remove chunk IDs from the global FAISS store if they are present."""
    if not ids:
        return

    path = vector_store_path()
    if not _has_saved_store(path):
        return

    store = _load_store(path, embeddings)
    existing_ids = set(store.index_to_docstore_id.values())
    ids_to_delete = [chunk_id for chunk_id in ids if chunk_id in existing_ids]
    if not ids_to_delete:
        return

    logger.info("从全局 FAISS 索引删除旧 chunk: %d", len(ids_to_delete))
    store.delete(ids_to_delete)
    store.save_local(str(path))


def build_and_save(
    *,
    repo_id: int,
    documents: Sequence[Document],
    ids: Sequence[str],
    embeddings: Embeddings,
) -> Path:
    """Embed `documents`, append them to the global FAISS index and persist it.

    Returns the directory where the index was saved.
    """
    if len(documents) != len(ids):
        raise ValueError("documents 和 ids 长度必须一致")
    if not documents:
        raise ValueError("没有可索引的 chunk")

    path = vector_store_path(repo_id)
    logger.info("写入全局 FAISS 索引 (repo_id=%d, chunk 数=%d)", repo_id, len(documents))
    if _has_saved_store(path):
        store = _load_store(path, embeddings)
        store.add_documents(documents=list(documents), ids=list(ids))
    else:
        store = FAISS.from_documents(documents=list(documents), embedding=embeddings, ids=list(ids))
    path.mkdir(parents=True, exist_ok=True)
    logger.info("保存 FAISS 索引到 %s", path)
    store.save_local(str(path))
    return path
