"""``metronome-pool`` console-script — build (and optionally pin) the held-out
eval pool.

* ``metronome-pool build --out <dir> [--sources openmeteo,wikimedia]`` —
  harvest real-world series, clean/validate them, and write the pool directory
  in the layout :mod:`metronome.validator.pool` reads back.
* add ``--upload`` to pack the directory, upload it to the Hippius registry, and
  print the ``[eval] window_pool`` CID to pin in ``chain.toml``. Requires the
  ``[hippius]`` extra + a reachable IPFS node.
* ``metronome-pool sources`` — list the registered sources.

Window geometry (``context_length`` / ``horizon``) defaults to ``[eval]`` in
``chain.toml`` so the pool matches what the validator expects; both are
overridable. Use ``--sources synthetic`` for an offline, network-free smoke test
of the whole path.

Exit codes: 0 = success, 1 = build produced no usable series, 2 = bad CLI usage,
4 = registry upload failure.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from ..shared.config import load_chain_config
from .builder import PoolBuildConfig, build_pool
from .source import HarvestContext, HttpFetcher
from .sources import DEFAULT_SOURCES, available, get_sources


def _parse_date(s: str | None) -> dt.date:
    if not s:
        return dt.date.today()
    return dt.date.fromisoformat(s)


def _add_build(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("build", help="Harvest real-world series into an eval-pool directory.")
    p.add_argument("--out", type=Path, required=True, help="Output pool directory.")
    p.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help=f"Comma-separated source names. Available: {', '.join(available())}.",
    )
    p.add_argument("--as-of", default=None, help="Freshness cutoff YYYY-MM-DD (default: today).")
    p.add_argument("--span-days", type=int, default=210, help="Recent history to request.")
    p.add_argument("--context-length", type=int, default=None, help="Override [eval] context_length.")
    p.add_argument("--horizon", type=int, default=None, help="Override [eval] horizon.")
    p.add_argument("--min-context", type=int, default=256, help="Minimum context a kept window affords.")
    p.add_argument("--max-missing-frac", type=float, default=0.2, help="Drop series gappier than this.")
    p.add_argument("--max-series-per-domain", type=int, default=None)
    p.add_argument("--max-series-total", type=int, default=None)
    p.add_argument("--max-series-per-source", type=int, default=10_000)
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--overwrite", action="store_true", help="Replace any existing pool at --out.")
    p.add_argument("--timeout", type=float, default=30.0, help="Per-request HTTP timeout (s).")
    p.add_argument(
        "--upload",
        action="store_true",
        help="Upload the built pool to the Hippius registry and print the CID to pin.",
    )
    p.set_defaults(func=_cmd_build)


def _add_sources(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("sources", help="List registered data sources.")
    p.set_defaults(func=_cmd_sources)


def _cmd_sources(args: argparse.Namespace) -> int:
    print("\n".join(available()))
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    cfg = load_chain_config(args.chain_toml)
    context_length = args.context_length or cfg.eval.context_length
    horizon = args.horizon or cfg.eval.horizon

    try:
        sources = get_sources([s.strip() for s in args.sources.split(",") if s.strip()])
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    ctx = HarvestContext(
        as_of=_parse_date(args.as_of),
        span_days=args.span_days,
        context_length=context_length,
        horizon=horizon,
        max_series=args.max_series_per_source,
    )
    build_cfg = PoolBuildConfig(
        context_length=context_length,
        horizon=horizon,
        min_context=args.min_context,
        max_missing_frac=args.max_missing_frac,
        max_series_per_domain=args.max_series_per_domain,
        max_series_total=args.max_series_total,
    )

    try:
        summary = build_pool(
            sources,
            args.out,
            ctx,
            build_cfg,
            fetch=HttpFetcher(timeout=args.timeout),
            overwrite=args.overwrite,
        )
    except (ValueError, FileExistsError) as e:
        print(f"build failed: {e}", file=sys.stderr)
        return 1

    print(summary.render())

    if summary.n_series < cfg.scoring.min_windows:
        print(
            f"warning: pool has {summary.n_series} series but [scoring] min_windows="
            f"{cfg.scoring.min_windows}; rounds may be inconclusive. Add sources/locations.",
            file=sys.stderr,
        )

    if args.upload:
        return _upload(args.out, cfg)
    print("\nnext: pin the pool with `--upload`, or upload the directory yourself and set")
    print("      [eval] window_pool = \"<cid>\" in chain.toml")
    return 0


def _upload(out_dir: Path, cfg) -> int:
    from ..shared.hippius import RegistryConfig, StorageError, upload_dir_to_registry

    try:
        reg = RegistryConfig.from_storage(cfg.storage)
        up = upload_dir_to_registry(out_dir, reg)
    except StorageError as e:
        print(f"registry upload failed: {e}", file=sys.stderr)
        return 4
    print(f"\nuploaded to Hippius registry: cid={up.cid} ({up.size_bytes:,} bytes)")
    print("pin this in chain.toml:")
    print(f'    [eval]\n    window_pool = "{up.cid}"')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="metronome-pool", description="Build the held-out eval pool for metronome validators."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_build(sub)
    _add_sources(sub)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
