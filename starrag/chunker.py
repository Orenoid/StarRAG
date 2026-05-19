"""Turn files into LangChain `Document` chunks.

For each file:
1. Read text (utf-8 with replacement on decoding errors).
2. Build a single `Document` with metadata (repo, owner, name, file_path,
   language).
3. Pick a `RecursiveCharacterTextSplitter`:
   - For languages supported by LangChain's `Language` enum, use
     `from_language(...)` so separators respect code structure.
   - Otherwise (plain text / json / yaml / unknown), use the default
     character-based splitter.
4. Split the document; chunks inherit the parent's metadata plus a
   `chunk_index`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


# Extension -> LangChain Language enum value.
# Source: docs.langchain.com/oss/python/integrations/splitters/code_splitter
_EXT_TO_LANGUAGE: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".kt": Language.KOTLIN,
    ".cpp": Language.CPP, ".cc": Language.CPP, ".cxx": Language.CPP, ".hpp": Language.CPP,
    ".c": Language.C, ".h": Language.C,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".swift": Language.SWIFT,
    ".cs": Language.CSHARP,
    ".scala": Language.SCALA,
    ".lua": Language.LUA,
    ".pl": Language.PERL,
    ".hs": Language.HASKELL,
    ".html": Language.HTML,
    ".md": Language.MARKDOWN, ".markdown": Language.MARKDOWN,
    ".rst": Language.RST,
    ".sol": Language.SOL,
    ".proto": Language.PROTO,
}


# Defaults — tuned for an MiniLM-class embedder (max seq ~256 tokens).
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Cache splitter instances; building one allocates regexes.
_splitter_cache: dict[str, RecursiveCharacterTextSplitter] = {}


def _get_splitter(ext: str) -> tuple[RecursiveCharacterTextSplitter, str]:
    """Return (splitter, language_tag) for the given extension."""
    lang = _EXT_TO_LANGUAGE.get(ext)
    if lang is not None:
        key = f"lang:{lang.value}"
        if key not in _splitter_cache:
            _splitter_cache[key] = RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        return _splitter_cache[key], lang.value
    # Fallback: plain recursive splitter (good enough for json/yaml/txt/etc.)
    if "default" not in _splitter_cache:
        _splitter_cache["default"] = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    return _splitter_cache["default"], "text"


def _read_file_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("读取文件失败,跳过 %s: %s", path, e)
        return None


def file_to_chunks(
    *,
    path: Path,
    repo_root: Path,
    repo_id: int,
    owner: str,
    name: str,
) -> list[Document]:
    """Read `path` and return its chunks as LangChain Documents."""
    text = _read_file_text(path)
    if text is None or not text.strip():
        return []

    rel_path = path.relative_to(repo_root).as_posix()
    ext = path.suffix.lower()
    splitter, language = _get_splitter(ext)

    parent = Document(
        page_content=text,
        metadata={
            "repo_id": repo_id,
            "owner": owner,
            "name": name,
            "file_path": rel_path,
            "extension": ext,
            "language": language,
        },
    )
    chunks = splitter.split_documents([parent])
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


def chunk_files(
    *,
    paths: Iterable[Path],
    repo_root: Path,
    repo_id: int,
    owner: str,
    name: str,
) -> Iterator[Document]:
    """Yield chunk Documents for every file in `paths`."""
    total_files = 0
    total_chunks = 0
    for path in paths:
        chunks = file_to_chunks(
            path=path,
            repo_root=repo_root,
            repo_id=repo_id,
            owner=owner,
            name=name,
        )
        if not chunks:
            continue
        total_files += 1
        total_chunks += len(chunks)
        for c in chunks:
            yield c
    logger.info("切分完成: %d 个文件 -> %d 个 chunk", total_files, total_chunks)
