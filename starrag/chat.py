"""`starrag chat` — simple REPL backed by the ``search_repo`` tool.

Loads the global FAISS index produced by `add`, then loops:
  1. read a query from stdin
  2. call search_repo(query) across all chunks
  3. print each hit's repo details and full chunk content
"""
from __future__ import annotations

import logging
import sqlite3

import click
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from . import db
from .tools import (
    DEFAULT_TOP_K,
    SearchRepoError,
    SearchRepoResult,
    load_global_store,
    search_repo,
)

logger = logging.getLogger(__name__)


TOP_K = DEFAULT_TOP_K


def _get_chunk_document(store: FAISS, chunk_id: str) -> Document | None:
    document = store.docstore.search(chunk_id)
    if isinstance(document, Document):
        return document
    logger.warning("FAISS docstore 中找不到 chunk_id=%s: %s", chunk_id, document)
    return None


def _print_hit(
    rank: int,
    result: SearchRepoResult,
    repo: sqlite3.Row | None,
    document: Document | None,
) -> None:
    repo_name = f"{repo['owner']}/{repo['name']}" if repo else "<unknown>"
    click.echo("")
    click.echo(
        click.style(
            f"#{rank}  [{repo_name}]  repoId={result['repoId']}  chunkId={result['chunkId']}",
            bold=True,
        )
    )
    if repo:
        click.echo(f"repo: {repo['url']}")

    if document is None:
        click.echo("(找不到 chunk 内容)")
        return

    metadata = document.metadata or {}
    file_path = metadata.get("file_path", "<unknown>")
    chunk_index = metadata.get("chunk_index", "?")
    language = metadata.get("language", "?")
    click.echo(f"file: {file_path}  [chunk {chunk_index}, {language}]")
    click.echo("-" * 80)
    click.echo(document.page_content)


def run_chat() -> None:
    logger.info("进入 chat 模式")
    try:
        store, repo_count = load_global_store()
    except SearchRepoError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("")
    click.echo(
        f"已加载 {repo_count} 个仓库的全局索引。输入查询后回车，"
        f"会打印全局 top-{TOP_K} 的 repo 信息和 chunk 内容。"
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
        try:
            hits = search_repo(query, top_k=TOP_K, store=store)
        except SearchRepoError as exc:
            click.echo(f"(搜索失败: {exc})")
            continue
        if not hits:
            click.echo("(没有命中任何 chunk)")
            continue
        repos: dict[int, sqlite3.Row | None] = {}
        for i, result in enumerate(hits, start=1):
            repo_id = result["repoId"]
            if repo_id not in repos:
                repos[repo_id] = db.get_repo_by_id(repo_id)
            document = _get_chunk_document(store, result["chunkId"])
            _print_hit(i, result, repos[repo_id], document)
