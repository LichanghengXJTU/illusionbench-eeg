"""stimuli/classical_cv/face_embedding.py — dlib face_recognition_resnet_v1.

Pre-trained 128-dim face embedding from dlib (Davis King, 2017). Used as a
single quantitative "identity similarity" metric for Part-Whole donor
matching: we want donors that are SIMILAR ENOUGH that the swap doesn't look
like a different ethnicity, but NOT SO similar that V1 ≈ V2.

Standard "same person" threshold in dlib: L2 < 0.6.
Our donor window: 0.55 < L2 < 0.85 (similar identity but not same person,
not jarringly different).
"""
from __future__ import annotations
import dlib
import numpy as np


class FaceEmbedder:
    """Loads dlib's HOG+SVM detector, 68-pt predictor, and 128-d face
    recognition ResNet. Computes embeddings from full RGB images.

    All three models are pre-trained, distributed by dlib (`davisking/dlib-models`),
    used purely as inference services (no fine-tuning)."""

    def __init__(self, predictor_path: str, recognizer_path: str):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.recognizer = dlib.face_recognition_model_v1(recognizer_path)

    def embed(self, rgb: np.ndarray) -> tuple[np.ndarray | None, "dlib.full_object_detection | None"]:
        """Detect single dominant face, return (128-d embedding, shape).
        Returns (None, None) if no face detected."""
        dets = self.detector(rgb, 1)
        if not dets:
            return None, None
        # Pick largest detection
        best = max(dets, key=lambda d: (d.right() - d.left()) * (d.bottom() - d.top()))
        shape = self.predictor(rgb, best)
        chip = dlib.get_face_chip(rgb, shape, size=150, padding=0.25)
        emb = self.recognizer.compute_face_descriptor(chip)
        return np.array(emb, dtype=np.float32), shape
