import click

from starrag.chat import run_chat
from starrag.ingest import add_repo
from starrag.logging_config import setup_logging


@click.group()
def cli():
    """StarRAG: clone a git repo and chat with it via RAG."""


@cli.command()
def chat():
    """在所有已添加的仓库里搜索，打印 top-10 repo 信息和 chunk 内容。"""
    setup_logging()
    run_chat()


@cli.command()
@click.argument("url")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="重新拉取并重建索引(即使仓库已存在)。",
)
def add(url: str, force: bool):
    """拉取 URL 指向的 git 仓库,切分、嵌入并存入 FAISS。"""
    setup_logging()
    add_repo(url, force=force)


if __name__ == "__main__":
    cli()
