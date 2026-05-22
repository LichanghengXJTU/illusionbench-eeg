# State

**Tick #**: 56
**Last updated**: 2026-05-22 ~15:45 (Asia/Hong_Kong)
**Current focus** (one sentence): Idea-003 / HOLO-Net — the full model failed to
train across 3 runs (v1/v2/v3); switched from patch-and-relaunch to a systematic
component-isolation ablation (E043) to find which component breaks identity.
**Last action**: Tick 56 — found v3 also failed (identity descended 13→10.6 then
rose to 14.1; orientation flat). Instrumented `train.py` with a `--enable` flag
for per-component ablation, killed v3, launched two parallel diagnostics.
**Last action outcome**: diag-A + diag-B running; descent trend readable next tick.
**Running tasks** (on server, H100 80GB — both training, 40 GB / 99% util):
  - **diag-A (E043)** `--enable ffa` — `/workspace/holo_net_diagA_ffa/`, minimal
    backbone + FFA holistic term, no aux losses. step ~550.
  - **diag-B (E043)** `--enable pc,ofa,gist,magno` —
    `/workspace/holo_net_diagB_aux/`, minimal backbone + magno + 3 aux losses,
    no FFA. step ~100.
**Stuck streak**: 0 (each tick has produced a tangible artifact / decision)
**Planned next action** (tick 57): read diag-A vs diag-B identity descent
(~step 4000+). Whichever group's identity fails to descend = the culprit;
narrow within it. If both descend → test `--enable orient` next. Goal: a
trainable full HOLO-Net.
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
