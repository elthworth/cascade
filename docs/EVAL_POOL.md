# Building the held-out eval pool

Validators score the king's and challenger's trained models on a **private,
rotating pool of real-world series** (`[eval] window_pool`, a Hippius registry
CID). This doc covers the **producer** side — turning real data into that pool —
with `metronome.pool` and the `metronome-pool` CLI. The consumer side (fetch CID
→ slice into windows) lives in `metronome.validator.pool` / `.windows`.

## Why this design is hard to game

The eval is only contamination-resistant if a miner's generator cannot
**distribution-match** the eval set and the trained model cannot **memorise** it.
The pool leans on three levers:

1. **Privacy** — owner-controlled, never published as a named public benchmark.
2. **Freshness** — sources harvest *recent* data up to an `as_of` cutoff; you
   re-build periodically so each pinned pool rotates in time. Data that didn't
   exist at submission can't be matched or memorised. (The validator also rotates
   the *slice* per round via the block hash.)
3. **Breadth** — multiple real domains at sub-daily frequency, so the only way to
   score well is to forecast generally, not to fit one distribution.

Do **not** point the pool at a fixed public benchmark (GIFT-Eval, Monash, …):
those are the easiest thing for a generator to overfit, and they overlap what
time-series foundation models pretrain on.

## Quick start

```bash
# Offline smoke test (no network): synthetic series through the full path.
metronome-pool build --out ./pool --sources synthetic --overwrite

# Real pool from the default sources (weather + web traffic), then pin it.
metronome-pool build --out ./pool --upload
# → prints:  [eval]
#            window_pool = "bafy…"     ← paste into chain.toml

metronome-pool sources   # list registered sources
```

Window geometry (`context_length` / `horizon`) defaults to `[eval]` in
`chain.toml`, so the pool matches what the validator expects.

## Sources (shipped)

| name        | domain      | freq | seasonality | notes |
|-------------|-------------|------|-------------|-------|
| `openmeteo` | weather     | H    | 24 (daily)  | keyless archive API; backbone, fills full context |
| `wikimedia` | web_traffic | D    | 7 (weekly)  | keyless REST API; shorter-context breadth |
| `synthetic` | synthetic   | H    | 24          | offline/testing only — **not** for a real pool |

The shipped location/article lists are a starting seed. **Scale the pool by
data, not code**: extend `LOCATIONS` / `VARIABLES` / `ARTICLES`, raise
`--max-series-*`, or add a source. Aim comfortably above `[scoring] min_windows`
(and ideally `[eval] n_windows`) — the CLI warns if the pool is too small.

## Adding a source

Implement the `DataSource` protocol (`metronome/pool/source.py`) — a `name` and
`harvest(fetch, ctx) -> Iterable[HarvestedSeries]` — and register it in
`metronome/pool/sources/__init__.py`. Do **all** network I/O through the injected
`fetch` callable so the source is unit-testable against canned JSON (see
`tests/unit/test_pool_sources.py`). Yield raw series with a pandas-style `freq`;
the builder handles cleaning, gap-fill, length normalisation, degeneracy/dup
filtering, and seasonality.

Good keyless candidates to add: ENTSO-E / EIA grid load (hourly), USGS water
services (hydrology, hourly), air-quality (OpenAQ). Crypto OHLCV is huge and
future-unknowable but near-random-walk, so use it as a minor domain only.

## What the builder guarantees

- **On-disk format** matches `metronome.validator.pool` exactly: one
  `<series_id>.npy` per series (float32; loader upcasts), `metadata.json` keyed by
  `series_id` → `{freq, seasonal_period, domain}`, plus a `provenance.json` the
  loader ignores.
- **Determinism**: same harvested inputs ⇒ byte-identical directory ⇒ identical
  Hippius CID (the registry packs a sorted, zeroed-metadata tar). Re-build to
  audit.
- **Cleaning**: gaps interpolated (series dropped above `--max-missing-frac`),
  truncated to the freshest `context_length + horizon` points, short/constant/
  multi-channel series dropped, exact duplicates de-duplicated.
