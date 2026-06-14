# metronome submission interface (for miners)

You submit a **data generator**, not a model. Your generator produces synthetic
univariate time-series that the subnet owner's trainer uses to train a fixed
forecasting model. You win when your data trains a better forecaster than the
king's data.

## Repo layout

Your HuggingFace repo must contain exactly:

```
generator.py        # exposes `class Generator(DataGenerator)`
config.json         # any JSON object; your generator may read it
requirements.txt    # hash-locked, allowlisted, <= max_packages
```

**No weight files.** `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `*.ckpt` are
rejected — the trainer produces weights, not you.

## The contract

```python
from collections.abc import Iterator
import numpy as np
from metronome.interface import DataGenerator

class Generator(DataGenerator):
    def __init__(self, config_dir: str, *, seed: int) -> None:
        # Load config_dir/config.json if you like. `seed` is your ONLY source
        # of randomness — derive everything from np.random.default_rng(seed).
        ...

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        # Yield EXACTLY n_series 1-D float arrays. Each length must fall in the
        # configured [min_length, max_length] band; total points are capped.
        ...

    @property
    def name(self) -> str:
        return "my-generator"
```

### Hard requirements

* **Determinism.** Two runs at the same `seed` must produce a byte-identical
  corpus. No wall-clock, no `os.urandom`, no un-seeded global RNG. `metronome
  verify` runs your generator twice and rejects it if the digests differ — this
  is non-negotiable, because the trainer and validators rely on it to audit
  runs.
* **Bounds.** Each series is finite (no NaN/inf), 1-D, floating dtype, with
  length in `[generator.min_length, generator.max_length]`. The whole corpus is
  capped at `generator.max_total_points`.
* **Count.** `generate(n)` yields exactly `n` series.
* **No network / no escape.** `generator.py` is AST-scanned for blocked imports
  (sockets, subprocess, the metronome internals, etc.) and run in a
  network-isolated sandbox. See `chain.toml [static_guard]`.
* **Dependencies.** `requirements.txt` lines must be `pkg==ver --hash=sha256:…`,
  drawn from `chain.toml [dependencies] allowed`, at most `max_packages`.

## Deploy

```bash
metronome verify ./my-generator-repo            # runs every trainer-side check
metronome deploy <org>/<repo> --revision <40-char-sha> \
    --wallet-name <coldkey> --wallet-hotkey <hotkey> --verify-dir ./my-generator-repo
```

`deploy` writes `metro-v1:gen:hf:<org>/<repo>@<sha>` on-chain via
`set_reveal_commitment`. The SHA pins the exact tree the trainer will fetch.

## What good data looks like

You're optimising for **downstream forecast generalisation** of a fixed model on
real held-out series (CRPS + MASE). Diversity of regimes (trend, multiple
seasonalities, regime shifts, varied noise structure, realistic scales) tends to
beat narrow or degenerate corpora. Memorising the eval set is not an option —
you never see it, and the trainer only ever feeds the model *your generator's
output*. See `scripts/example_generator/` for a runnable starting point.
