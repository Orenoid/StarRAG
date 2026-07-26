"""Reusable tools exposed by StarRAG."""
from __future__ import annotations

import logging
from typing import TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from . import db
from .embeddings import build_embeddings
from .vectorstore import vector_store_path

logger = logging.getLogger(__name__)


DEFAULT_TOP_K = 10


class SearchRepoResult(TypedDict):
    """A chunk matched by :func:`search_repo`."""

    chunkId: str
    repoId: int


class SearchRepoError(RuntimeError):
    """Raised when repo search cannot be completed."""


def load_global_store() -> tuple[FAISS, int]:
    """Load the shared FAISS store and return it with the tracked repo count."""
    db.init_schema()
    repos = db.list_repos()
    if not repos:
        raise SearchRepoError("没有已添加的仓库。先运行 `starrag add <url>` 添加。")

    path = vector_store_path()
    if not (path / "index.faiss").exists() or not (path / "index.pkl").exists():
        raise SearchRepoError("没有可用的全局索引。请运行 `starrag add <url>` 添加或重建仓库。")

    store = FAISS.load_local(
        str(path),
        build_embeddings(),
        allow_dangerous_deserialization=True,
    )
    logger.info("加载全局 FAISS 索引: %s (repos=%d)", path, len(repos))
    return store, len(repos)


def _result_from_document(document: Document) -> SearchRepoResult:
    metadata = document.metadata or {}
    repo_id = metadata.get("repo_id")
    if repo_id is None:
        raise SearchRepoError("搜索结果缺少 repo_id，请重建对应仓库的索引。")

    chunk_id = metadata.get("chunk_id")
    if not chunk_id:
        chunk_id = db.get_chunk_id(
            repo_id=int(repo_id),
            file_path=str(metadata.get("file_path", "")),
            chunk_index=int(metadata.get("chunk_index", 0)),
        )
    if not chunk_id:
        raise SearchRepoError(
            f"搜索结果缺少 chunk_id（repo_id={repo_id}），请重建对应仓库的索引。"
        )

    return {"chunkId": str(chunk_id), "repoId": int(repo_id)}


def search_repo(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    store: FAISS | None = None,
) -> list[SearchRepoResult]:
    """Search all indexed repos and return the top-k chunk/repo ID pairs.

    ``store`` is an optional dependency-injection parameter. Callers handling
    multiple queries can load the global store once and reuse it.
    """
    query = query.strip()
    if not query:
        raise ValueError("query 不能为空")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    if store is None:
        store, _ = load_global_store()

    hits = store.similarity_search_with_score(query, k=top_k)
    hits.sort(key=lambda hit: hit[1])
    results = [_result_from_document(document) for document, _ in hits[:top_k]]
    logger.info("向量搜索完成: top_k=%d, hits=%d", top_k, len(results))
    return results
