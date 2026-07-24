# custom-fullctx-v4 — all-4096 target-efficiency (the real-budget win)

v3's de-trended (length-normalized bimodal trend) families, emitting ONLY
full-context 4096-pt series (`min_length = max_length = 4096`).

**Mechanism.** The trainer buckets each series into `P = min(L//32, 128)` patches
and yields `P−1` next-patch targets, so total targets `= BUDGET/32 − N`. Fewer,
longer series → more learning per token AND training at the exact ctx-4096 eval
geometry (128 patches). At the real budget, distributional diversity saturates
(v2 already streams ~390k fresh series), so this target-efficiency is what binds.

**Result (real testnet budget 500M tok, ctx 4096, production-faithful streaming,
5 seeds, honest clustering): v4 vs v2 = +3.39%, LCB +0.0205, 5/5 seeds — robust
WIN.** Tight per-seed spread (+2.3…+4.7%). Numpy-only (2.5M tok/s, 13× the wall),
deterministic, matmul-free. v2 cannot copy: its trend blows up at 4096.
