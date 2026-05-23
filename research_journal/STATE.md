# State

**Tick #**: 71
**Last updated**: 2026-05-23 ~09:30 (Asia/Hong_Kong)
**Current focus** (one sentence): User picked FLAG-002 option 4 — diagnostic
first; wrote `HOLONET_DIAGNOSTIC.md` (literature-grounded per-component
critique + a HOLO-Net v2.0 design sketch).
**Last action**: Tick 71 — web-searched the actual Thatcher neural mechanism
(Psalta et al. 2014; Taubert 2015 monkey; Boutsen 2006 N170 ERP); identified
3 concrete bio-mechanism implementation gaps in v1.0 (Thatcher mechanism =
wrong locus; missing fSTS analogue; FFA self-attention lacks the second-
order relational primitive); sketched HOLO-Net v2.0 — 3 targeted revisions
(local orientation-tuned feature module; fSTS-style upright-grotesqueness
branch; Capsule-style routing-by-agreement FFA).
**Last action outcome**: diagnostic done. Leading reading: implementation
gap (A) is real and primary; training-objective (B) is also plausible but
not yet cleanly tested without a properly bio-faithful architecture.
**Running tasks** (on server, H100 80GB): none.
**Stuck streak**: 0
**Planned next action** (tick 72): await user confirmation to build v2.0
(~1 day work + 3 h training). If confirmed → implement
LocalOrientationModule + fSTS branch + Capsule FFA, train, evaluate. If user
prefers a different v2.0 scope, adjust.
**Cadence**: ~50 min fallback heartbeat; user is driving.
**Confidence in current best idea**:
  - **Idea-001** (IllusionBench-EEG): **9.3/10** — unaffected, and strengthened:
    the HOLO-Net negative result is architecture-side corroboration of its
    training-objective thesis.
  - **Idea-003** (HOLO-Net): **5.0/10** — the pre-registered "first
    paradigm-consistent model" ambition is refuted; architecture trains fine;
    a redirect (objective re-aim, or consolidate as a mechanistic negative)
    is open.

## The HOLO-Net result in one line
A maximally bio-faithful face architecture, trained on face identity, shows NO
Thatcher / composite illusion (ISI/CSI ≈ 1.0) — exactly like FaceNet (E004).
Bio-fidelity of architecture does not substitute for the training objective.

## Strategic note (user directive 2026-05-22)
Idea-003 (HOLO-Net) was designated the headline. The v5 refutation means that,
as a positive headline, HOLO-Net v1.0 does not deliver — see FLAG-002 for the
three honest options. Loop continues; full autonomy on direction; will not stop.

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/` (plain dir, not git)
- HOLO-Net: `holo_net_stage2/` (minimal/E040-E041), `holo_net_full_v5/`
  (full, trained, E042/E044); v1-v4 dead
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
