"""Submission validation: generator commit format, requirements, repo layout."""

from __future__ import annotations

import pytest

from metronome.interface.validation import (
    check_repo_layout,
    check_requirements_hash_locked,
    format_commit,
    parse_commit,
)

GOOD_SHA = "abc123def456abc123def456abc123def456abcd"


def test_parse_commit_round_trip():
    payload = f"metro-v1:gen:hf:foo-org/some_repo@{GOOD_SHA}"
    parsed = parse_commit(payload)
    assert parsed is not None
    assert parsed.repo == "foo-org/some_repo"
    assert parsed.revision == GOOD_SHA
    assert format_commit("foo-org/some_repo", GOOD_SHA) == payload


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "metro-v0:gen:hf:foo/bar@" + GOOD_SHA,
        "metro-v1:trained:hf:foo/bar@" + GOOD_SHA,  # trained tag is not a gen commit
        "metro-v1:gen:gcs:foo/bar@" + GOOD_SHA,
        "metro-v1:gen:hf:foo@" + GOOD_SHA,
        "metro-v1:gen:hf:foo/bar",
        "metro-v1:gen:hf:foo/bar@deadbeef",
        f"metro-v1:gen:hf:foo/bar@{GOOD_SHA}extra",
        f"metro-v1:gen:hf: foo/bar@{GOOD_SHA}",
        "metro-v1:gen:hf:/bar@" + GOOD_SHA,
    ],
)
def test_parse_commit_rejects_malformed(payload):
    assert parse_commit(payload) is None


def test_format_commit_refuses_invalid_inputs():
    with pytest.raises(ValueError):
        format_commit("no_slash_in_repo", GOOD_SHA)
    with pytest.raises(ValueError):
        format_commit("foo/bar", "not-a-sha")


def test_parse_commit_normalises_uppercase_sha():
    parsed = parse_commit(f"metro-v1:gen:hf:foo/bar@{GOOD_SHA.upper()}")
    assert parsed is not None
    assert parsed.revision == GOOD_SHA


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return p


def test_repo_layout_accepts_generator_repo(tmp_path):
    _write(tmp_path, "config.json", "{}")
    _write(tmp_path, "generator.py", "x = 1\n")
    _write(tmp_path, "requirements.txt", "numpy==1.26.4 --hash=sha256:" + "a" * 64 + "\n")
    assert check_repo_layout(tmp_path).ok


def test_repo_layout_rejects_weight_files(tmp_path):
    _write(tmp_path, "config.json", "{}")
    _write(tmp_path, "generator.py", "x = 1\n")
    _write(tmp_path, "requirements.txt", "")
    _write(tmp_path, "model.safetensors", "binary")
    r = check_repo_layout(tmp_path)
    assert not r.ok
    assert r.reason == "weight_files_forbidden"


def test_repo_layout_rejects_missing_files(tmp_path):
    _write(tmp_path, "config.json", "{}")
    r = check_repo_layout(tmp_path)
    assert not r.ok
    assert r.reason == "missing_files"


def test_requirements_hash_locked(tmp_path):
    ok = _write(tmp_path, "ok.txt", f"numpy==1.26.4 --hash=sha256:{'a' * 64}\n")
    assert check_requirements_hash_locked(ok, allowed=("numpy",), max_packages=5).ok

    unpinned = _write(tmp_path, "bad.txt", "numpy>=1.0\n")
    r = check_requirements_hash_locked(unpinned, allowed=("numpy",), max_packages=5)
    assert not r.ok and r.reason == "requirement_not_hash_locked"

    bad_pkg = _write(tmp_path, "bad2.txt", f"evil==1.0 --hash=sha256:{'b' * 64}\n")
    r = check_requirements_hash_locked(bad_pkg, allowed=("numpy",), max_packages=5)
    assert not r.ok and r.reason == "requirement_not_allowlisted"
