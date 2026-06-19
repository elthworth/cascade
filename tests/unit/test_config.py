"""chain.toml loads and exposes the expected schema."""

from __future__ import annotations

from metronome.eval.koth import KothParams


def test_config_loads(cfg):
    assert cfg.schema_version == 1
    assert cfg.subnet.name == "metronome"
    assert cfg.generator.corpus_n_series > 0
    assert cfg.generator.min_length < cfg.generator.max_length
    assert cfg.training.epochs >= 1
    assert cfg.eval.n_windows > 0
    assert cfg.scoring.dethrone_cp >= 1


def test_koth_params_builds_from_scoring(cfg):
    params = cfg.koth_params()
    assert isinstance(params, KothParams)
    assert params.win_margin_start <= params.win_margin_end
    assert params.dethrone_cp == cfg.scoring.dethrone_cp


def test_static_guard_blocks_internal_modules(cfg):
    blocked = cfg.static_guard.blocked
    assert "metronome.trainer" in blocked
    assert "metronome.shared.chain" in blocked
    assert "socket" in blocked


def test_generator_allowlist_excludes_torch(cfg):
    # Generators emit data; they must not need torch.
    assert "torch" not in {a.lower() for a in cfg.dependencies.allowed}
