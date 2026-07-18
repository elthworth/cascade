---
id: DEC-CA-0002
type: decision
title: "Mainnet home: Bittensor netuid 91"
status: active
date: 2026-07-14
tags: [chain, launch, operations]
revisit_when: "mainnet launch complete — checklist in docs/MAINNET_LAUNCH.md fully cleared"
relations: {}
---
cascade launches on Bittensor mainnet netuid 91 (decided 2026-07-14). The
full pre-mainnet checklist lives in `docs/MAINNET_LAUNCH.md`; the blockers
are the L40S pin, the rebuilt worker-image digest, the container sandbox, and
the pool-publish cron, with the gift gate launching shadow → enforce. The
shipped `chain.toml` keeps `netuid = 0` as a template placeholder; the
mainnet deployment sets `[subnet] netuid = 91` alongside the other operator
inputs (`base_arch_digest`, `ref_throughput_tokens_per_s`, Hippius
`[storage]` credentials, `[eval] window_pool`). Throughput policy for the
contract is [[DEC-CA-0001]].
