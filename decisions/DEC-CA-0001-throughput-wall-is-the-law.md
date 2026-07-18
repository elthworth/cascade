---
id: DEC-CA-0001
type: decision
title: "Throughput policy: wall is the law"
status: active
date: 2026-07-14
tags: [training, incentives, throughput]
revisit_when: "enough duel telemetry to judge whether throughput dominates data quality in throne outcomes (starvation telemetry in the S3 log sink logs/round-<id>/<role>-<size>.jsonl; per-round roll-ups once the telemetry PR lands)"
relations: {}
---
`ref_throughput_tokens_per_s` (185k) is deliberately calibrated to a WELL-FED
trainer on the reference GPU, not to the median miner pipeline (~80k measured
across 554 testnet runs). Combined with `max_train_seconds` = the budget
hours, this makes generator throughput a compute MULTIPLIER: a miner's
realized training compute is proportional to their pipeline speed (a median
generator trains ~43% of the token budget before the wall stops it;
`deadline_hit` in the run record). This is intentional — the competition
prices data quality × generation speed, and slow generation self-penalizes;
do NOT "fix" mass deadline_hits by loosening the wall or shrinking the
budget. The alternative ("budget is the law": set ref to the measured median
so equal-compute duels are the norm and speed is only a floor) is one config
line + coordinated trainer+validator restart at a boundary — `[training]`
keys fold into `contract_digest`. See [[DEC-CA-0002]] and
[[NOTE-ca-operational-invariants]].
