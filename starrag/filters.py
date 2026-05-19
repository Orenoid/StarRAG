"""File walker + exclusion rules.

Strategy ported from deepwiki-open (`api/config.py`):
- A list of directory names that are always skipped (venv, node_modules, build, ...).
- A list of file names / glob patterns that are always skipped (lock files,
  binaries, minified bundles, ...).
- An allowlist of extensions for which we actually produce chunks, split into
  `CODE_EXTENSIONS` (source) and `DOC_EXTENSIONS` (markdown/text/config).
- A file size cap to skip anything pathological (huge generated files).
"""
from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


DEFAULT_EXCLUDED_DIRS: set[str] = {
    # Virtual environments and package managers
    ".venv", "venv", "env", "virtualenv",
    "node_modules", "bower_components", "jspm_packages",
    # Version control
    ".git", ".svn", ".hg", ".bzr",
    # Cache and compiled files
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".coverage",
    # Build and distribution
    "dist", "build", "out", "target", "bin", "obj",
    # Documentation build artifacts (raw docs/ source we DO want — keep separate)
    "_site", "site-docs", "_docs",
    # IDE specific
    ".idea", ".vscode", ".vs", ".eclipse", ".settings",
    # Logs and temporary files
    "logs", "log", "tmp", "temp",
}


DEFAULT_EXCLUDED_FILE_PATTERNS: list[str] = [
    # Lock files
    "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json", "poetry.lock",
    "Pipfile.lock", "requirements.txt.lock", "Cargo.lock", "composer.lock",
    "*.lock",
    # OS / IDE junk
    ".DS_Store", "Thumbs.db", "desktop.ini", "*.lnk",
    # Env / config noise
    ".env", ".env.*", "*.env", "*.cfg", "*.ini", ".flaskenv",
    # Git / CI / linter config
    ".gitignore", ".gitattributes", ".gitmodules", ".gitlab-ci.yml",
    ".prettierrc", ".eslintrc*", ".eslintignore", ".stylelintrc",
    ".editorconfig", ".jshintrc", ".pylintrc", ".flake8", "mypy.ini",
    # Build / bundler config
    "webpack.config.js", "babel.config.js", "rollup.config.js",
    "jest.config.js", "karma.conf.js", "vite.config.js", "next.config.js",
    # Minified / bundled / source-map
    "*.min.js", "*.min.css", "*.bundle.js", "*.bundle.css", "*.map",
    # Archives / binaries
    "*.gz", "*.zip", "*.tar", "*.tgz", "*.rar", "*.7z", "*.iso",
    "*.dmg", "*.img", "*.msix", "*.appx", "*.appxbundle", "*.xap", "*.ipa",
    "*.deb", "*.rpm", "*.msi", "*.exe", "*.dll", "*.so", "*.dylib",
    "*.o", "*.obj", "*.jar", "*.war", "*.ear", "*.jsm", "*.class",
    "*.pyc", "*.pyd", "*.pyo", "*.a", "*.lib", "*.lo", "*.la", "*.slo",
    # Egg / dist info
    "*.egg", "*.egg-info", "*.dist-info",
]


# Extensions we will chunk + embed.
CODE_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".kt",
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php", ".swift", ".cs", ".scala",
    ".sh", ".bash", ".zsh", ".lua", ".pl", ".hs",
    ".html", ".css", ".scss", ".vue", ".svelte",
}

DOC_EXTENSIONS: set[str] = {
    ".md", ".markdown", ".txt", ".rst",
    ".json", ".yaml", ".yml", ".toml",
}

ALLOWED_EXTENSIONS: set[str] = CODE_EXTENSIONS | DOC_EXTENSIONS


# Hard cap on file size to skip generated / vendored monoliths.
# Code: ~1 MB ought to cover anything sane.
# Doc:  ~256 KB — beyond that it's almost certainly generated.
MAX_CODE_FILE_BYTES = 1 * 1024 * 1024
MAX_DOC_FILE_BYTES = 256 * 1024


def _is_excluded_filename(name: str) -> bool:
    for pattern in DEFAULT_EXCLUDED_FILE_PATTERNS:
        if fnmatch.fnmatch(name, pattern):
            return True
    return False


def _path_has_excluded_dir(rel_path: Path) -> bool:
    return any(part in DEFAULT_EXCLUDED_DIRS for part in rel_path.parts)


def iter_repo_files(repo_root: Path) -> Iterator[Path]:
    """Yield absolute paths of files in `repo_root` that pass all filters.

    Logs the count of kept / skipped files at the end.
    """
    repo_root = repo_root.resolve()
    kept = 0
    skipped_dir = 0
    skipped_name = 0
    skipped_ext = 0
    skipped_size = 0

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root)

        if _path_has_excluded_dir(rel):
            skipped_dir += 1
            continue
        if _is_excluded_filename(path.name):
            skipped_name += 1
            continue

        ext = path.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            skipped_ext += 1
            continue

        try:
            size = path.stat().st_size
        except OSError:
            skipped_size += 1
            continue

        cap = MAX_CODE_FILE_BYTES if ext in CODE_EXTENSIONS else MAX_DOC_FILE_BYTES
        if size > cap:
            logger.debug("跳过过大文件 (%d 字节): %s", size, rel)
            skipped_size += 1
            continue

        kept += 1
        yield path

    logger.info(
        "文件遍历完成: 保留 %d / 跳过 [目录黑名单=%d, 文件名=%d, 扩展名=%d, 过大或损坏=%d]",
        kept, skipped_dir, skipped_name, skipped_ext, skipped_size,
    )
