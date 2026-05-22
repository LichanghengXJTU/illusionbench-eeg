# State

**Tick #**: 59
**Last updated**: 2026-05-23 ~01:35 (Asia/Hong_Kong)
**Current focus** (one sentence): Idea-003 / HOLO-Net — v4 (full model) training
through its normal bounce; tick 59 completed the 3-paradigm ablation baseline
and validated the eval pipeline for composite + part-whole.
**Last action**: Tick 59 — resolved the "CSI/PWI metric code" question (it IS
`compute_metrics.py` — the composite/part-whole generators reuse the V1-V4
condition slots). Ran the minimal checkpoint on composite + part-whole → the
full ablation floor (E041).
**Last action outcome**: minimal HOLO-Net (FFA layer) fails the falsification on
ALL 3 paradigms — Thatcher ISI ≈1.0, Composite CSI ≈1.19, Part-Whole PWI ≈0.78
(all corrected; thresholds 3.0 / 1.5 / 0.5). Intended ablation floor confirmed.
**Running tasks** (on server, H100 80GB):
  - **v4 — full HOLO-Net (E042 / Q017)** — `/workspace/holo_net_full_v4/`,
    step ~6300/30000, identity 10.65 (tracking the minimal run's bounce:
    minimal step 6000 ≈ 11.58). 0.37 s/step, ~2.4 h to completion.
**Stuck streak**: 0
**Planned next action** (tick 60): monitor v4 (~step 10000 — minimal there was
9.1, descent starting). Judge v4 properly at step ~15000+. When v4 finishes →
3-paradigm per-layer eval = the HOLO-Net falsification test.
**Confidence in current best idea**:
  - **Idea-003 (HOLO-Net)**: ~7/10 — eval pipeline + ablation floor now fully
    in place; the decisive v4 verdict is ~1 h away.
  - **Idea-001**: 9.3/10 — groundwork, stable.

## Watch-item (for v4 verdict)
v4 `orientation` loss flat at ~0.695 through step 6300 (same as v3). Under the
additive wiring (`atl_input = afp_pooled + gate·ffa_out`), the identity loss can
minimise by driving the gate→const (clean afp_pooled), which competes with the
orientation CE loss. If v4's identity descends but orientation stays flat, the
OrientationGate is inert → the Thatcher mechanism is weakened → a real design
issue to address AFTER the Q017 identity verdict.

## Resolved / done
- CSI & PWI metric code: it is `compute_metrics.py` on the composite/partwhole
  manifests — no new code needed. 3-paradigm eval pipeline validated.
- Minimal ablation baseline: complete for all 3 paradigms (E041).

## Strategic note (user directive 2026-05-22)
Idea-001 is theoretical groundwork; **Idea-003 (HOLO-Net) is the headline and
the loop's primary focus**. Full autonomy on Idea-003 direction. If compute
becomes the blocker, write a NEED to `NOTES_FOR_USER.md`. Keep looping;
commit+push and report status every tick.

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/` (plain dir, not git)
- HOLO-Net: `holo_net_stage2/` (minimal/E040-E041, done), `holo_net_full_v4/`
  (full/E042/Q017, running)
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
