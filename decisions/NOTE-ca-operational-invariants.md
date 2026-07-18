---
id: NOTE-ca-operational-invariants
type: note
title: "Operational invariants (hard-learned)"
status: active
date: 2026-07-14
tags: [operations, deployment]
relations: {}
---
Hard-learned operational rules for running cascade; violating any of these
has bitten us before.

- `[training]` edits change `contract_digest` → the VALIDATOR must restart
  too, or it rejects every manifest (`contract_digest_mismatch`). This is
  also why the [[DEC-CA-0001]] alternative needs a coordinated restart at a
  boundary.
- Pods are rsync'd trees, not git checkouts; `uv sync` needs `--all-extras`
  (torch lives behind the `train` extra).
- Never restart the provisioner inside its pre-boundary trigger window.
