"""Schema + runtime checks for a submitted generator repo.

Used by the miner CLI (``metronome verify``) and by the trainer before it
imports and runs a generator in the sandbox. Unlike horizon, a metronome
submission carries NO model weights — it is pure generator code plus a pinned
dependency set.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# A generator HF repo must contain exactly these files (weights are forbidden —
# the trainer produces weights, not the miner).
REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
    "generator.py",
    "requirements.txt",
)

# Weight-file globs that must NOT appear in a generator repo. Their presence is
# a strong signal the miner confused metronome (data) with horizon (models).
FORBIDDEN_WEIGHT_GLOBS: tuple[str, ...] = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt")

# requirements.txt line: ``pkg==1.2.3 --hash=sha256:abc...`` (one or more hash flags).
_REQ_LINE = re.compile(
    r"""
    ^
    (?P<name>[A-Za-z0-9_.\-]+)        # package name
    \s*==\s*                          # ==
    (?P<version>[A-Za-z0-9_.\-+]+)    # version
    (?P<hashes>(\s+--hash=sha256:[A-Fa-f0-9]{64})+)
    \s*(\#.*)?$
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str | None = None
    details: dict | None = None

    @classmethod
    def fail(cls, reason: str, **details) -> ValidationResult:
        return cls(ok=False, reason=reason, details=details or None)

    @classmethod
    def pass_(cls) -> ValidationResult:
        return cls(ok=True)


def check_repo_layout(repo_dir: Path | str) -> ValidationResult:
    """Required files present and no stray weight files."""
    d = Path(repo_dir)
    if not d.is_dir():
        return ValidationResult.fail("not_a_directory", path=str(d))
    missing = [name for name in REQUIRED_FILES if not (d / name).is_file()]
    if missing:
        return ValidationResult.fail("missing_files", missing=missing)
    weights = [p.name for g in FORBIDDEN_WEIGHT_GLOBS for p in d.glob(g)]
    if weights:
        return ValidationResult.fail("weight_files_forbidden", files=sorted(weights))
    return ValidationResult.pass_()


def check_config(repo_dir: Path | str) -> ValidationResult:
    """``config.json`` is present and parses as a JSON object."""
    d = Path(repo_dir)
    config_p = d / "config.json"
    if not config_p.is_file():
        return ValidationResult.fail("missing_config_json")
    try:
        obj = json.loads(config_p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return ValidationResult.fail("config_json_invalid", error=str(e))
    if not isinstance(obj, dict):
        return ValidationResult.fail("config_json_not_object")
    return ValidationResult.pass_()


def check_requirements_hash_locked(
    requirements_path: Path | str,
    allowed: tuple[str, ...] | None,
    max_packages: int,
) -> ValidationResult:
    """Reject if any line isn't hash-pinned, count exceeds limit, or a package
    name is outside the allowlist (when an allowlist is supplied)."""
    p = Path(requirements_path)
    if not p.is_file():
        return ValidationResult.fail("missing_requirements")
    lines = [
        ln.strip()
        for ln in p.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    # Join continuation lines (``\``-terminated) — pip allows them.
    joined: list[str] = []
    buf = ""
    for ln in lines:
        if ln.endswith("\\"):
            buf += ln[:-1] + " "
        else:
            joined.append(buf + ln)
            buf = ""
    if buf:
        joined.append(buf)

    if len(joined) > max_packages:
        return ValidationResult.fail(
            "too_many_packages", count=len(joined), max=max_packages
        )

    names = []
    for ln in joined:
        m = _REQ_LINE.match(ln)
        if not m:
            return ValidationResult.fail("requirement_not_hash_locked", line=ln)
        names.append(m.group("name").lower())

    if allowed is not None:
        allow_set = {n.lower() for n in allowed}
        bad = [n for n in names if n not in allow_set]
        if bad:
            return ValidationResult.fail("requirement_not_allowlisted", packages=bad)

    return ValidationResult.pass_()


# ----- on-chain commit format --------------------------------------------------

# Single pointer string: ``metro-v1:gen:hf:<org>/<repo>@<git_sha>``
# git_sha is 40-char hex (full SHA-1). The ``gen`` tag distinguishes a miner's
# generator submission from the trainer's ``trained`` pointers in the manifest.
COMMIT_RE = re.compile(
    r"^metro-v1:gen:hf:(?P<repo>[A-Za-z0-9][A-Za-z0-9._\-]*/[A-Za-z0-9][A-Za-z0-9._\-]*)"
    r"@(?P<sha>[A-Fa-f0-9]{40})$"
)


@dataclass(frozen=True)
class ParsedCommit:
    repo: str
    revision: str


def parse_commit(payload: str) -> ParsedCommit | None:
    """Return None for malformed payloads. The trainer treats None as a
    permanent rejection of the submission.

    Mixed-case SHAs are accepted and normalised to lowercase so identity
    checks downstream are case-stable.
    """
    m = COMMIT_RE.match(payload.strip())
    if not m:
        return None
    return ParsedCommit(repo=m.group("repo"), revision=m.group("sha").lower())


def format_commit(repo: str, revision: str) -> str:
    """Build the on-chain payload. Raises if the inputs would not round-trip
    through :func:`parse_commit`. The revision is lowercased so two callers
    that differ only in SHA case produce identical payloads.
    """
    payload = f"metro-v1:gen:hf:{repo}@{revision.lower()}"
    if parse_commit(payload) is None:
        raise ValueError(f"refusing to emit malformed commit: {payload!r}")
    return payload
