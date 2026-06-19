"""The reference generator runs, is deterministic, and passes verify."""

from __future__ import annotations

from dataclasses import replace

import pytest

from metronome.miner.verify import verify_repo
from metronome.trainer.corpus import assert_corpus_reproducible, build_corpus


@pytest.fixture()
def small_cfg(cfg):
    """Shrink the series count so the python AR(1) loop runs fast under test.
    The length band is left at the chain.toml values the example generator's
    own config.json (128..1024) fits inside."""
    gen = replace(cfg.generator, corpus_n_series=6)
    return replace(cfg, generator=gen)


def test_example_generator_builds_corpus(small_cfg, example_generator_dir):
    res = build_corpus(example_generator_dir, generation_seed=0, cfg=small_cfg.generator)
    assert res.n_series == 6
    assert res.total_points > 0
    assert len(res.digest) == 64


def test_example_generator_is_deterministic(small_cfg, example_generator_dir):
    d = assert_corpus_reproducible(example_generator_dir, 0, small_cfg.generator)
    assert len(d) == 64
    # Different seed → different corpus.
    other = build_corpus(example_generator_dir, generation_seed=1, cfg=small_cfg.generator)
    assert other.digest != d


def test_verify_accepts_example_generator(small_cfg, example_generator_dir):
    report = verify_repo(example_generator_dir, small_cfg, skip_runtime=False)
    assert report.ok, report.render()
    assert report.corpus_digest is not None


def test_verify_static_path_only(small_cfg, example_generator_dir):
    report = verify_repo(example_generator_dir, small_cfg, skip_runtime=True)
    assert report.ok
    assert report.runtime_skipped
