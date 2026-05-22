# State

**Tick #**: 53
**Last updated**: 2026-05-22 ~13:45 (Asia/Hong_Kong)
**Current focus** (one sentence): Idea-003 / HOLO-Net — the full architecture is
training (E042); this tick evaluated the minimal-mode checkpoint to establish
the "minus all bio components" ablation baseline and validate the eval harness.
**Last action**: Tick 53 — ran `eval_extract.py` + `compute_metrics.py` on the
minimal checkpoint (step 25000). Created E041.
**Last action outcome**: CONFIRMED — minimal HOLO-Net Thatcher ISI ≈ 1.0 at all
7 layers (0.965-1.037); no Thatcher effect without the bio components. Eval
pipeline validated end-to-end (v1 ISI 1.019 ≈ pixel baseline).
**Running tasks** (on server, H100 80GB):
  - PID 44488 — minimal run (E040 / ablation baseline), step ~26K/30K,
    identity ~1.8, ETA ~30 min. Checkpoint frozen at step 25000.
  - PID 47222 — **FULL HOLO-Net (E042)**, step ~4K/30K, identity ~13 (margin
    warmup just ended at step 4000 — real descent expected to begin now),
    predcode 0.12, face_detect 0.06, gist 0.16. **orientation 0.71 — flat (Q015)**.
**Stuck streak**: 0
**Planned next action** (tick 54): monitor the full run — confirm identity-loss
descent now that margin warmup is done, and watch the Orientation Gate (Q015).
When the minimal run finishes → re-eval its final checkpoint. Locate the
composite-CSI / part-whole-PWI metric code so all 3 paradigms can be scored.
**Confidence in current best idea**:
  - **Idea-003 (HOLO-Net)**: ~7.5/10 — architecture trains; ablation floor now
    established (minimal ISI≈1); awaiting the full-model falsification result.
  - **Idea-001**: 9.3/10 — groundwork, stable.

## Open watch-items
- **Q015**: full-run `orientation` loss flat at chance (≈0.69) through step 3850.
  Re-check at step ~10000; if still flat → real bug → the Thatcher mechanism
  (design §7) would be defeated → fix + relaunch E042.

## Strategic note (user directive 2026-05-22)
Idea-001 is theoretical groundwork; **Idea-003 (HOLO-Net) is the headline and
the loop's primary focus**. Full autonomy on Idea-003 direction. Build / train /
validate HOLO-Net continuously. If compute becomes the blocker, write a NEED to
`NOTES_FOR_USER.md` — the user will rent more GPU. Keep looping; commit+push and
report status every tick.

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/` (plain dir, not git)
- HOLO-Net runs: `/workspace/holo_net_stage2/` (minimal/E040),
  `/workspace/holo_net_full_v1/` (full/E042)
- Stimulus manifests (server): `data/stimuli_ffhq{,_composite,_partwhole,_randombbox}/thatcher_manifest.csv`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
