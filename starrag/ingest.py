"""Orchestrator for the `starrag add <url>` command.

Steps:
1. Ensure data dirs + sqlite schema exist.
2. Bail out if the repo was already added.
3. Shallow-clone the repo into data/repos.
4. Insert a row in `repos`, get its id.
5. Walk the repo with the filter rules from `filters`.
6. Chunk every file into LangChain Documents (`chunker`).
7. Assign each chunk a uuid id, build embeddings, save FAISS index.
8. Persist the chunk metadata rows in sqlite so we can trace a vector_id
   back to (repo, file, chunk_index).
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from . import db
from .chunker import chunk_files
from .embeddings import build_embeddings
from .filters import iter_repo_files
from .git_utils import clone_repo
from .paths import ensure_dirs
from .vectorstore import build_and_save, vector_store_path

logger = logging.getLogger(__name__)


def _preview(text: str, limit: int = 200) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:limit]


def add_repo(url: str, *, force: bool = False) -> None:
    """End-to-end ingestion for a single git repo URL."""
    logger.info("=" * 60)
    logger.info("开始添加仓库: %s", url)
    logger.info("=" * 60)

    ensure_dirs()
    db.init_schema()

    # ---- 1. skip-if-exists guard --------------------------------------------------
    existing = db.get_repo_by_url(url)
    if existing and not force:
        logger.info(
            "仓库已存在 (repo_id=%d, %s/%s),如需重建请加 --force",
            existing["id"], existing["owner"], existing["name"],
        )
        return

    # ---- 2. clone -----------------------------------------------------------------
    logger.info("[1/5] 拉取代码")
    local_path, owner, name = clone_repo(url, force=force)

    # ---- 3. record / refresh repo row --------------------------------------------
    logger.info("[2/5] 写入 SQLite 元数据")
    if existing and force:
        repo_id = int(existing["id"])
        db.delete_repo_chunks(repo_id)
        old_vector_dir = Path(existing["vector_store_path"])
        if old_vector_dir.exists():
            logger.info("清理旧 FAISS 索引: %s", old_vector_dir)
            shutil.rmtree(old_vector_dir, ignore_errors=True)
        # We don't update the row itself (URL is the same); the path columns
        # were already correct.
        v_path = vector_store_path(repo_id)
    else:
        # We need repo_id BEFORE we know vector_store_path layout; since the
        # path is purely derived from repo_id, we insert with a placeholder
        # and update once we have it.
        repo_id = db.insert_repo(
            url=url,
            owner=owner,
            name=name,
            local_path=local_path,
            vector_store_path=Path("pending"),  # filled in below
        )
        v_path = vector_store_path(repo_id)
        with db.connect() as conn:
            conn.execute(
                "UPDATE repos SET vector_store_path = ? WHERE id = ?",
                (str(v_path), repo_id),
            )
        logger.info("repo_id=%d", repo_id)

    # ---- 4. walk + chunk ----------------------------------------------------------
    logger.info("[3/5] 遍历文件并切分 chunk")
    files = list(iter_repo_files(local_path))
    if not files:
        logger.warning("没有可处理的文件,流程结束")
        return

    chunks = list(
        chunk_files(
            paths=files,
            repo_root=local_path,
            repo_id=repo_id,
            owner=owner,
            name=name,
        )
    )
    if not chunks:
        logger.warning("切分后无 chunk,流程结束")
        return

    chunk_ids = [str(uuid.uuid4()) for _ in chunks]

    # ---- 5. embed + persist FAISS -------------------------------------------------
    logger.info("[4/5] 构建嵌入并写入 FAISS")
    embeddings = build_embeddings()
    build_and_save(
        repo_id=repo_id,
        documents=chunks,
        ids=chunk_ids,
        embeddings=embeddings,
    )

    # ---- 6. persist chunk metadata -----------------------------------------------
    logger.info("[5/5] 写入 chunk 元数据到 SQLite")
    rows = []
    for cid, doc in zip(chunk_ids, chunks):
        meta = doc.metadata
        rows.append(
            {
                "id": cid,
                "repo_id": repo_id,
                "file_path": meta.get("file_path", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "language": meta.get("language"),
                "content_preview": _preview(doc.page_content),
                "char_count": len(doc.page_content),
            }
        )
    db.insert_chunks(rows)

    logger.info("=" * 60)
    logger.info("仓库添加完成: %s/%s (repo_id=%d, chunks=%d)", owner, name, repo_id, len(chunks))
    logger.info("=" * 60)
