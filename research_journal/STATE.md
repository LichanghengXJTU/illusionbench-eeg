# State

**Tick #**: 57
**Last updated**: 2026-05-23 ~00:20 (Asia/Hong_Kong)
**Current focus** (one sentence): Idea-003 / HOLO-Net — component-isolation
ablation (E043) running to find which component breaks full-model identity
training; diag-A/B inconclusive so far, diag-D added.
**Last action**: Tick 57 — read diag-A/diag-B at step ~3000 (both descend
13→10.6, similar, neither catastrophic, neither past the step-4000 margin
transition where v3 blew up). Launched diag-D (`--enable ffa,orient`) to test
the gate×FFA interaction.
**Last action outcome**: inconclusive yet — decisive window (step 3000-5000) not
reached; diag-D added to cover the orient×FFA hypothesis.
**Running tasks** (on server, H100 80GB — 3 runs, 60 GB):
  - **diag-A (E043)** `--enable ffa` — step ~3050, identity ~10.6.
  - **diag-B (E043)** `--enable pc,ofa,gist,magno` — step ~2600, identity ~10.6.
  - **diag-D (E043)** `--enable ffa,orient` — step ~50, identity 13.3.
**Stuck streak**: 0 (each tick has produced a tangible artifact / decision)
**Planned next action** (tick 58): read diag-A/B/D past step ~4000-5000 — the
window where v3's identity blew up. Whichever config's identity stays down vs
blows up isolates the culprit. Goal: a trainable full HOLO-Net.
**Confidence in current best idea**:
  - **Idea-003 (HOLO-Net)**: ~6.5/10 — lowered: the full architecture has
    resisted training across 3 runs. The ablation (E043) will say whether it is
    a fixable single component or a deeper multi-task design problem.
  - **Idea-001**: 9.3/10 — groundwork, stable, untouched.

## Note on progress
Ticks 51-56 have been HOLO-Net debugging: 3 real bugs found+fixed (AdaFace loss
formula; OrientationGate flip-blindness; FFA/ATL identity-collapse), but the
full model still does not train. E043 is the systematic isolation that should
have come before the patch attempts. If E043 shows no single fixable culprit,
consider: (a) curriculum init from the minimal checkpoint, (b) longer AdaFace
margin warmup, (c) staged aux-loss introduction.

## Known minor issues (non-blocking, deferred)
- OFA / PFC-Gist face-vs-nonface heads see only faces (`is_face` always 1 — the
  design's 10% ImageNet non-face mix is not implemented in `data.py`).
- Deferred since tick 53: re-eval minimal FINAL checkpoint; locate composite-CSI
  / part-whole-PWI metric code.

## Strategic note (user directive 2026-05-22)
Idea-001 is theoretical groundwork; **Idea-003 (HOLO-Net) is the headline and
the loop's primary focus**. Full autonomy on Idea-003 direction. If compute
becomes the blocker, write a NEED to `NOTES_FOR_USER.md`. Keep looping;
commit+push and report status every tick.

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/` (plain dir, not git)
- HOLO-Net: `holo_net_stage2/` (minimal/E040, done), `holo_net_diagA_ffa/` +
  `holo_net_diagB_aux/` (E043); full runs v1/v2/v3 dead
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
