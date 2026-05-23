"""EEG decoder Stage 3 — frozen DINOv2 target via per-subject ridge mapping
from ATM-style EEG features.

Design lock: research_journal/EEG_DECODER_STAGE3_DESIGN.md (tick 81).

Submodules:
  - data.py: load ATM EEG features + image features (THINGS-EEG2)
  - extract_dinov2_things.py: extract DINOv2 features on THINGS images (tick 82)
  - stage3_ridge.py: train per-subject ridge ATM-EEG → DINOv2 (tick 83)
  - illusionbench_transfer.py: IllusionBench-EEG transfer eval (tick 84)
"""
