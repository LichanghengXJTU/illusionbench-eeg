"""HOLO-Net: A strict bio-fidelity face-vision model with each layer mapped to
a specific brain region. Designed to emerge paradigm-consistent holistic face
illusion sensitivity from clean identity training (no Thatcher exposure).

Brain region → component mapping:
  LGN-Magno/Parvo → dual input streams
  V1 / V2 / V4 / IT(MFP) → CORnet-S backbone
  OFA → face/non-face branch off V4
  AFP → view-invariance contrastive block
  Orientation Gate → upright/inverted soft gating before FFA
  FFA → 3-step recurrent self-attention with Magno-gist global query
  ATL → AdaFace identity head (512-d L2-normalized embedding)
  PFC-Gist → magno → MLP → gist_token (Bar 2003 top-down)
  PC feedback → custom V1↔V2↔V4↔IT predictive coding loops

See research_journal/IDEA_003_HOLO_NET_DESIGN.md for full design rationale.
"""
from .model import HOLONet, HOLONetConfig

__all__ = ["HOLONet", "HOLONetConfig"]
