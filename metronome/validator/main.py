"""``metronome-validator`` console-script entry point.

Runs the round loop indefinitely against the live chain. ``--offline`` builds
the runner and prints the loaded champion state without any chain or HF I/O — a
plumbing smoke check.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .loop import build_runner


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="metronome-validator", description="metronome validator loop.")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--network", default="finney", help="Bittensor network (finney/test/local).")
    p.add_argument("--wallet-name", default=None)
    p.add_argument("--wallet-hotkey", default=None)
    p.add_argument("--wallet-path", default=None)
    p.add_argument("--hf-cache-dir", type=Path, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--offline", action="store_true", help="No chain/HF; print state and exit.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    runner = build_runner(
        chain_toml=args.chain_toml,
        hf_cache_dir=args.hf_cache_dir,
        device=args.device,
    )

    if args.offline:
        print(f"netuid:   {runner.cfg.netuid}")
        print(f"king:     {runner.state.king_hotkey}")
        print(f"tenure:   {runner.state.tenure_rounds}")
        print(f"dethrone_cp: {runner.cfg.scoring.dethrone_cp}")
        print("offline validator smoke complete")
        return 0

    if args.wallet_name is None or args.wallet_hotkey is None:
        print("--wallet-name and --wallet-hotkey are required unless --offline", file=sys.stderr)
        return 2

    # TODO: live loop — poll the manifest repo on an interval, process_round on
    # each new round, set winner-take-all weights on runner.state.king_uid, and
    # persist state to [validator] state_db_path. Left as a boundary so the
    # chain + HF integration is reviewed on its own.
    print("live validator loop not yet wired; see TODO in metronome/validator/main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
