"""Git helpers: extract repo name from URL and shallow clone."""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

from .paths import REPOS_DIR, ensure_dirs

logger = logging.getLogger(__name__)


_URL_RE = re.compile(r"^(?:https?://|git@)")


def parse_repo_name(url: str) -> tuple[str, str]:
    """Return (owner, name) from a git URL.

    Supports common forms:
      - https://github.com/owner/repo
      - https://github.com/owner/repo.git
      - git@github.com:owner/repo.git
      - https://gitlab.com/group/subgroup/repo (uses last two segments)
    """
    cleaned = url.strip().rstrip("/")
    if cleaned.startswith("git@"):
        cleaned = cleaned.split(":", 1)[1]
    elif "://" in cleaned:
        cleaned = cleaned.split("://", 1)[1].split("/", 1)[1]
    if cleaned.endswith(".git"):
        cleaned = cleaned[: -len(".git")]
    parts = [p for p in cleaned.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"无法从 URL 解析 owner/name: {url}")
    return parts[-2], parts[-1]


def ensure_git_installed() -> None:
    try:
        subprocess.run(
            ["git", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError("未检测到 git CLI,请先安装 git") from e


def clone_repo(url: str, *, force: bool = False) -> tuple[Path, str, str]:
    """Shallow-clone `url` into REPOS_DIR/{owner}_{name}.

    Returns (local_path, owner, name). If the target dir already exists
    and is non-empty, reuse it unless `force=True`.
    """
    ensure_dirs()
    ensure_git_installed()

    owner, name = parse_repo_name(url)
    target = REPOS_DIR / f"{owner}_{name}"

    if target.exists() and any(target.iterdir()):
        if force:
            logger.info("强制重新拉取,删除已有目录: %s", target)
            shutil.rmtree(target)
        else:
            logger.info("目录已存在且非空,复用现有代码: %s", target)
            return target, owner, name

    target.mkdir(parents=True, exist_ok=True)
    logger.info("开始 git clone (depth=1): %s -> %s", url, target)
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth=1",
                "--single-branch",
                url,
                str(target),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        # Cleanup the (probably empty) target dir if clone failed
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        stderr = e.stderr.decode("utf-8", "replace") if e.stderr else ""
        raise RuntimeError(f"git clone 失败: {stderr.strip()}") from e
    logger.info("git clone 完成")
    return target, owner, name
