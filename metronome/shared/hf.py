"""HuggingFace helpers — fetch (miner/validator) and upload (trainer).

* ``fetch_revision`` resolves a pinned ``(repo, revision)`` to a local
  directory. Used by ``metronome verify`` (generator repo), by the trainer
  (materialise a generator before sandboxed corpus generation), and by the
  validator (pull a trained checkpoint named in the manifest).
* ``upload_folder`` / ``upload_manifest`` are the trainer-side pushes. They are
  thin wrappers over ``huggingface_hub`` kept behind a lazy import so the rest
  of the package stays installable without it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SHA_RE = re.compile(r"^[a-f0-9]{40}$")


@dataclass(frozen=True)
class FetchResult:
    repo: str
    revision: str
    local_dir: Path


class FetchError(RuntimeError):
    """HF resolution or download failed."""


def fetch_revision(
    repo: str,
    revision: str,
    *,
    cache_dir: Path | str | None = None,
    token: str | None = None,
) -> FetchResult:
    """Materialise ``repo@revision`` to a local directory at the exact SHA.

    Raises :class:`FetchError` on any failure so the caller can mark the
    submission transiently invalid and retry next round.
    """
    revision = revision.lower()
    if not _SHA_RE.match(revision):
        raise FetchError(f"revision not a 40-char lowercase hex SHA: {revision!r}")
    try:
        from huggingface_hub import snapshot_download  # type: ignore
        from huggingface_hub.utils import (  # type: ignore
            EntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError as e:
        raise FetchError(f"huggingface_hub not installed: {e}") from e

    try:
        path = snapshot_download(
            repo_id=repo,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            token=token,
        )
    except RepositoryNotFoundError as e:
        raise FetchError(f"repo_not_found: {repo}") from e
    except RevisionNotFoundError as e:
        raise FetchError(f"revision_not_found: {repo}@{revision}") from e
    except EntryNotFoundError as e:
        raise FetchError(f"entry_not_found: {e}") from e
    except Exception as e:  # noqa: BLE001
        raise FetchError(f"snapshot_download_failed: {type(e).__name__}: {e}") from e

    local = Path(path)
    if not local.is_dir():
        raise FetchError(f"snapshot path is not a directory: {local}")
    return FetchResult(repo=repo, revision=revision, local_dir=local)


def fetch_manifest_text(
    repo: str,
    filename: str,
    *,
    revision: str = "main",
    cache_dir: Path | str | None = None,
    token: str | None = None,
) -> str:
    """Download a single manifest file from the owner dataset repo and return
    its text. Used by the validator to read the current training manifest."""
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as e:
        raise FetchError(f"huggingface_hub not installed: {e}") from e
    try:
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            token=token,
        )
    except Exception as e:  # noqa: BLE001
        raise FetchError(f"manifest_download_failed: {type(e).__name__}: {e}") from e
    return Path(path).read_text(encoding="utf-8")


def upload_folder(
    local_dir: Path | str,
    repo: str,
    *,
    token: str | None = None,
    commit_message: str = "metronome trainer: trained checkpoint",
) -> str:
    """Trainer-side: push a trained checkpoint folder and return the commit SHA.

    Thin wrapper over ``HfApi.upload_folder``. Raises :class:`FetchError` on
    failure. The returned 40-char SHA is what the trainer pins into the
    manifest via :func:`metronome.shared.manifest.format_trained_pointer`.
    """
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError as e:
        raise FetchError(f"huggingface_hub not installed: {e}") from e
    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo, exist_ok=True)
        commit = api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo,
            commit_message=commit_message,
        )
    except Exception as e:  # noqa: BLE001
        raise FetchError(f"upload_folder_failed: {type(e).__name__}: {e}") from e
    # upload_folder returns a CommitInfo; oid is the commit SHA.
    return str(getattr(commit, "oid", commit))
