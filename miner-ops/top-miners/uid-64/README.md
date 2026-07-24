# custom-longctx-lrfam-v5 — all-4096 + long-range-family reweight

v4's all-4096 target-efficiency PLUS a family reweight toward long-range-structure
priors (integrated random-walk, regime-shift, persistent AR2, GP, multi-seasonal)
and away from choppy short-range ones (chaotic/intermittent/pulse) that waste the
128-patch context. De-trended (from v3). Hypothesis: at a 4096-token context the
model benefits from long-range learnable structure. Under test at the real budget
against v4/v2. Numpy-only, deterministic.
