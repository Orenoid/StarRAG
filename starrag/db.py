"""SQLite metadata store.

Records each cloned repo and every chunk produced from it so the FAISS
vectors can be traced back to their source repo + file.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .paths import DB_PATH, ensure_dirs

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL UNIQUE,
    owner TEXT NOT NULL,
    name TEXT NOT NULL,
    local_path TEXT NOT NULL,
    vector_store_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    repo_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    language TEXT,
    content_preview TEXT,
    char_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(repo_id) REFERENCES repos(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_repo_file ON chunks(repo_id, file_path);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema() -> None:
    logger.info("初始化 SQLite schema: %s", DB_PATH)
    with connect() as conn:
        conn.executescript(_SCHEMA)


def get_repo_by_url(url: str) -> sqlite3.Row | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM repos WHERE url = ?", (url,)).fetchone()
    return row


def get_repo_by_id(repo_id: int) -> sqlite3.Row | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM repos WHERE id = ?", (repo_id,)).fetchone()
    return row


def list_repos() -> list[sqlite3.Row]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM repos ORDER BY created_at DESC").fetchall()
    return list(rows)


def insert_repo(
    *, url: str, owner: str, name: str, local_path: Path, vector_store_path: Path
) -> int:
    logger.info("写入 repo 元数据: %s/%s", owner, name)
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO repos (url, owner, name, local_path, vector_store_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (url, owner, name, str(local_path), str(vector_store_path)),
        )
        return int(cur.lastrowid)


def delete_repo_chunks(repo_id: int) -> None:
    logger.info("清理 repo_id=%d 的旧 chunk 记录", repo_id)
    with connect() as conn:
        conn.execute("DELETE FROM chunks WHERE repo_id = ?", (repo_id,))


def list_chunk_ids_by_repo(repo_id: int) -> list[str]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id FROM chunks WHERE repo_id = ? ORDER BY created_at",
            (repo_id,),
        ).fetchall()
    return [str(row["id"]) for row in rows]


def get_chunk_id(*, repo_id: int, file_path: str, chunk_index: int) -> str | None:
    """Resolve a chunk ID from the metadata stored in older FAISS indexes."""
    with connect() as conn:
        row = conn.execute(
            """
            SELECT id
            FROM chunks
            WHERE repo_id = ? AND file_path = ? AND chunk_index = ?
            LIMIT 1
            """,
            (repo_id, file_path, chunk_index),
        ).fetchone()
    return str(row["id"]) if row else None


def insert_chunks(rows: list[dict]) -> None:
    if not rows:
        return
    logger.info("写入 %d 条 chunk 元数据", len(rows))
    with connect() as conn:
        conn.executemany(
            """
            INSERT INTO chunks
                (id, repo_id, file_path, chunk_index, language, content_preview, char_count)
            VALUES
                (:id, :repo_id, :file_path, :chunk_index, :language, :content_preview, :char_count)
            """,
            rows,
        )
