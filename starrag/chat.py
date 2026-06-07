"""`starrag chat` — simple REPL that searches one global chunk index.

Loads the global FAISS index produced by `add`, then loops:
  1. read a query from stdin
  2. run similarity_search_with_score(query, k=10) across all chunks
  3. print each hit's repo, file_path, chunk_index, score, and a content preview
"""
from __future__ import annotations

import logging

import click
from langchain_community.vectorstores import FAISS

from . import db
from .embeddings import build_embeddings
from .vectorstore import vector_store_path

logger = logging.getLogger(__name__)


TOP_K = 10
PREVIEW_CHARS = 300


def _load_global_store() -> tuple[FAISS, int]:
    """Return the shared FAISS store and the number of repos tracked in SQLite."""
    repos = db.list_repos()
    if not repos:
        raise click.ClickException("没有已添加的仓库。先运行 `starrag add <url>` 添加。")

    v_path = vector_store_path()
    if not (v_path / "index.faiss").exists() or not (v_path / "index.pkl").exists():
        raise click.ClickException("没有可用的全局索引。请运行 `starrag add <url>` 添加或重建仓库。")

    embeddings = build_embeddings()
    store = FAISS.load_local(
        str(v_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("加载全局 FAISS 索引: %s (repos=%d)", v_path, len(repos))
    return store, len(repos)


def _search_all(store: FAISS, query: str) -> list[tuple]:
    """Run query against the global store, sort by score (asc), return top-K.

    Each result is a tuple: (score, doc, repo_owner, repo_name).
    """
    all_hits: list[tuple[float, object, str, str]] = []
    hits = store.similarity_search_with_score(query, k=TOP_K)
    for doc, score in hits:
        meta = doc.metadata or {}
        all_hits.append((score, doc, meta.get("owner", "?"), meta.get("name", "?")))
    all_hits.sort(key=lambda t: t[0])
    return all_hits[:TOP_K]


def _print_hit(rank: int, doc, score: float, owner: str, name: str) -> None:
    meta = doc.metadata or {}
    file_path = meta.get("file_path", "<unknown>")
    chunk_index = meta.get("chunk_index", "?")
    language = meta.get("language", "?")
    content = doc.page_content.strip()
    if len(content) > PREVIEW_CHARS:
        content = content[:PREVIEW_CHARS].rstrip() + " …"
    click.echo("")
    click.echo(
        click.style(
            f"#{rank}  score={score:.4f}  [{owner}/{name}]  {file_path}  [chunk {chunk_index}, {language}]",
            bold=True,
        )
    )
    click.echo("-" * 80)
    click.echo(content)


def run_chat() -> None:
    logger.info("进入 chat 模式")
    store, repo_count = _load_global_store()
    click.echo("")
    click.echo(
        f"已加载 {repo_count} 个仓库的全局索引。输入查询后回车,全局 top-{TOP_K} 结果会被打印。"
    )
    click.echo("空行或输入 exit/quit 退出。")

    while True:
        try:
            query = click.prompt("\n> ", prompt_suffix="", default="", show_default=False)
        except (EOFError, click.Abort):
            click.echo("")
            break
        query = query.strip()
        if not query or query.lower() in {"exit", "quit"}:
            break

        logger.info("查询: %s", query)
        hits = _search_all(store, query)
        if not hits:
            click.echo("(没有命中任何 chunk)")
            continue
        for i, (score, doc, owner, name) in enumerate(hits, start=1):
            _print_hit(i, doc, score, owner, name)
