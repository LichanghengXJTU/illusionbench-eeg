"""stimuli/classical_cv/insight_attrs.py — InsightFace gender+age extraction.

Why a wrapper:
  Direct InsightFace `app.get(rgb_1024)` fails on most FFHQ portraits because
  the face fills the frame and the detector's anchors are tuned for smaller
  faces in scene context. We bypass this by using the dlib face bbox we
  already have, cropping with 50% padding, resizing to 400x400, then running
  InsightFace's detection + genderage at det_size=(320,320).

Verified yield: 9/9 detect on hand-picked FFHQ; gender ~89% accurate.
"""
from __future__ import annotations
import numpy as np
from PIL import Image


class InsightAttrExtractor:
    """Thread-local-safe wrapper. One instance per worker process."""

    def __init__(self):
        # Force every ORT InferenceSession in this process to single-threaded.
        # Without this, each worker spawns ~core_count threads per session and
        # 64+ workers thrash the scheduler (Phase A v1 stuck at zero progress).
        import onnxruntime as ort
        _orig_init = ort.InferenceSession.__init__
        def _patched(self, path_or_bytes, sess_options=None, *a, **kw):
            if sess_options is None:
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 1
                sess_options.inter_op_num_threads = 1
            return _orig_init(self, path_or_bytes, sess_options, *a, **kw)
        ort.InferenceSession.__init__ = _patched
        import insightface
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "genderage"],
        )
        # ctx_id=-1 → CPU
        self.app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.3)

    @staticmethod
    def crop_with_pad(rgb: np.ndarray, lm: np.ndarray,
                       pad_frac: float = 0.5,
                       out_size: int = 400) -> np.ndarray:
        """Crop face using dlib landmark bbox, expand by pad_frac, resize."""
        h, w = rgb.shape[:2]
        x1, y1 = lm[:, 0].min(), lm[:, 1].min()
        x2, y2 = lm[:, 0].max(), lm[:, 1].max()
        face_w = x2 - x1
        face_h = y2 - y1
        pw = face_w * pad_frac
        ph = face_h * pad_frac
        x1 = max(0, int(x1 - pw)); y1 = max(0, int(y1 - ph))
        x2 = min(w, int(x2 + pw)); y2 = min(h, int(y2 + ph))
        crop = rgb[y1:y2, x1:x2]
        return np.array(Image.fromarray(crop).resize(
            (out_size, out_size), Image.LANCZOS))

    def predict(self, rgb: np.ndarray, lm: np.ndarray
                ) -> tuple[int, str] | None:
        """Return (age, gender) where gender ∈ {'M', 'F'}, or None if
        no face detected."""
        crop = self.crop_with_pad(rgb, lm)
        # InsightFace expects BGR
        faces = self.app.get(crop[..., ::-1])
        if not faces:
            return None
        f = max(faces, key=lambda x: x.det_score)
        return (int(f.age), "M" if f.gender == 1 else "F")
