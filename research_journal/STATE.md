# State

**Tick #**: 68
**Last updated**: 2026-05-23 ~07:14 (Asia/Hong_Kong)
**Current focus** (one sentence): Continuing paper integration — added the
HOLO-Net §4.7 "architecture test" subsection to PAPER_DRAFT; ABSTRACT next.
**Last action**: Tick 68 — wrote **§4.7 Architecture test** in `PAPER_DRAFT.md`
(~470 words) — the constructive-negative subsection: HOLO-Net pre-registered
falsification, the FFA-layer numbers, the FFA-not-inert sub-finding, and the
conclusion that the operative lever is the training objective (Claim 8). No
user steer on FLAG-002 yet.
**Last action outcome**: paper now has the architecture-test result inline
with the other Results subsections.
**Running tasks** (on server, H100 80GB): none.
**Stuck streak**: 0
**Planned next action** (tick 69): update `ABSTRACT.md` (v4) to add a sentence
on the HOLO-Net constructive-negative; also update `TLDR_FOR_PI.md` to reflect
the architecture-test outcome. If FLAG-002 steered → act on it.
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
