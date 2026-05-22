# State

**Tick #**: 50
**Last updated**: 2026-05-22 ~12:20 (Asia/Hong_Kong)
**Current focus** (one sentence): HOLO-Net (Idea-003) Stage-2 face fine-tune is
running in a de-risked MINIMAL MODE after the full bio-fidelity architecture
failed to optimize; this tick = check on training + journal resync.
**Last action**: Tick 50 — polled server; synced `holo_net/` code (local↔server
verified identical); created E040 documenting the full-model training failure;
added Q014; flagged the HOLO-Net training failure to the user (FLAG-001 in
NOTES_FOR_USER.md).
**Last action outcome**: no-result-yet (training in progress). Sub-results:
full-model training failure CONFIRMED; minimal-mode training CONFIRMED working.
**Running tasks** (on server):
  - PID 44488 — HOLO-Net Stage-2 minimal-mode training. step ~15850/30000 (~53%),
    identity loss 13.1→5.8, view-invariance 0.70→0.09, ETA ~85-90 min.
    Log: `/workspace/holo_net_stage2/train_log.jsonl`
**Stuck streak**: 0 (tick produced E040 + Q014 + journal resync)
**Planned next action** (tick 51): poll training. If done → E041, evaluate the
minimal checkpoint on the 3 IllusionBench paradigms + random-bbox (a sub-path (a)
result). If still running → re-check and reschedule. Then Q014 component-ablation
ladder to isolate the full-model failure.
**Confidence in current best idea**:
  - **Idea-001** (IllusionBench-EEG paper): **9.3/10** — paper-ready, unchanged.
    This is the safe, publishable deliverable.
  - **Idea-003** (HOLO-Net): **~6.5/10** — down from 8.5. The full bio-fidelity
    architecture does not train; the current run only tests sub-path (a).

## Journal hygiene note
Ticks ~44-49 debugged Stage-2 training but did NOT update STATE.md / DECISIONS.md
per tick (loop constraint 7 was violated during that span). Tick 50 reconstructs
that period from server artifacts (failed-run logs + checkpoints). STATE.md must
be rewritten every tick from here on.

## Working directories
- Local Mac: `~/Desktop/EEG/illusionbench/`
- Server: `/workspace/illusionbench-eeg/`  (NOTE: not a git repo — plain dir)
- HOLO-Net training output: `/workspace/holo_net_stage2/`
- ATM repo: `/workspace/eeg_repos/EEG_Image_decode/`
- Server SSH: `ssh -i ~/.ssh/id_ed25519 -p 11022 root@103.207.149.173`
- GitHub: https://github.com/LichanghengXJTU/illusionbench-eeg
