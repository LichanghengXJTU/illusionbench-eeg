"""models/registry.py — model loaders for IllusionBench-EEG.

Each loader returns (embed_fn, info, model_obj). embed_fn takes a sequence of PIL.Image
and returns an L2-normalized (N, D) float32 numpy array on CPU.

Convention: every embedding is L2-normalized so that cosine distance = 1 - dot product.
"""
from __future__ import annotations

import gc
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------
# CLIP family — HuggingFace transformers
# ---------------------------------------------------------------------

def load_clip_hf(hf_id: str, device: str = "cuda", dtype=torch.float16):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(hf_id, torch_dtype=dtype).to(device).eval()
    processor = CLIPProcessor.from_pretrained(hf_id)

    def embed(images: Sequence[Image.Image]) -> np.ndarray:
        inputs = processor(images=list(images), return_tensors="pt", padding=True).to(device)
        # cast pixel_values to model dtype if needed
        if inputs["pixel_values"].dtype != dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        with torch.no_grad():
            feats = model.get_image_features(pixel_values=inputs["pixel_values"])
            feats = F.normalize(feats, dim=-1)
        return feats.cpu().float().numpy()

    info = {"hf_id": hf_id, "output_dim": int(model.config.projection_dim)}
    return embed, info, model


# ---------------------------------------------------------------------
# SigLIP / SigLIP2 — image-text contrastive with sigmoid loss (Q007)
# ---------------------------------------------------------------------

def load_siglip_hf(hf_id: str, device: str = "cuda", dtype=torch.float16):
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(hf_id, torch_dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(hf_id)

    def embed(images: Sequence[Image.Image]) -> np.ndarray:
        inputs = processor(images=list(images), return_tensors="pt").to(device)
        if inputs["pixel_values"].dtype != dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        with torch.no_grad():
            feats = model.get_image_features(pixel_values=inputs["pixel_values"])
            feats = F.normalize(feats, dim=-1)
        return feats.cpu().float().numpy()

    out_dim = int(model.config.vision_config.hidden_size)
    info = {"hf_id": hf_id, "output_dim": out_dim}
    return embed, info, model


# ---------------------------------------------------------------------
# MetaCLIP — same CLIPModel API, different data curation (Q007)
# ---------------------------------------------------------------------

def load_metaclip_hf(hf_id: str, device: str = "cuda", dtype=torch.float16):
    # MetaCLIP uses the same architecture as CLIP — load via CLIPModel
    return load_clip_hf(hf_id, device=device, dtype=dtype)


# ---------------------------------------------------------------------
# DINOv2 — HuggingFace transformers
# ---------------------------------------------------------------------

def load_dinov2_hf(hf_id: str, device: str = "cuda", dtype=torch.float16):
    from transformers import AutoImageProcessor, AutoModel
    model = AutoModel.from_pretrained(hf_id, torch_dtype=dtype).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(hf_id)

    def embed(images: Sequence[Image.Image]) -> np.ndarray:
        inputs = processor(images=list(images), return_tensors="pt").to(device)
        if inputs["pixel_values"].dtype != dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        with torch.no_grad():
            out = model(**inputs)
            feats = out.last_hidden_state[:, 0, :]  # CLS token
            feats = F.normalize(feats, dim=-1)
        return feats.cpu().float().numpy()

    info = {"hf_id": hf_id, "output_dim": int(model.config.hidden_size)}
    return embed, info, model


# ---------------------------------------------------------------------
# MAE — HuggingFace transformers, encoder only (mask_ratio=0)
# ---------------------------------------------------------------------

def load_mae_hf(hf_id: str, device: str = "cuda", dtype=torch.float32):
    from transformers import AutoImageProcessor, ViTMAEModel
    model = ViTMAEModel.from_pretrained(hf_id, torch_dtype=dtype).to(device).eval()
    model.config.mask_ratio = 0.0
    processor = AutoImageProcessor.from_pretrained(hf_id)

    def embed(images: Sequence[Image.Image]) -> np.ndarray:
        inputs = processor(images=list(images), return_tensors="pt").to(device)
        if inputs["pixel_values"].dtype != dtype:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        with torch.no_grad():
            out = model(**inputs)
            # mean-pool over patch tokens (after CLS)
            hidden = out.last_hidden_state  # (B, 1+N_patches, D) with mask_ratio=0
            feats = hidden.mean(dim=1)
            feats = F.normalize(feats, dim=-1)
        return feats.cpu().float().numpy()

    info = {"hf_id": hf_id, "output_dim": int(model.config.hidden_size)}
    return embed, info, model


# ---------------------------------------------------------------------
# SDXL VAE — diffusers
# ---------------------------------------------------------------------

def load_sdxl_vae(hf_id: str = "madebyollin/sdxl-vae-fp16-fix",
                  device: str = "cuda", target_size: int = 512):
    from diffusers import AutoencoderKL
    import torchvision.transforms as T
    model = AutoencoderKL.from_pretrained(hf_id, torch_dtype=torch.float16).to(device).eval()
    transform = T.Compose([
        T.Resize(target_size, antialias=True),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
    ])

    def embed(images: Sequence[Image.Image]) -> np.ndarray:
        batch = torch.stack([transform(img) for img in images]).to(device).half()
        with torch.no_grad():
            latent = model.encode(batch).latent_dist.mean  # (B, 4, H/8, W/8)
        feats = latent.flatten(1)
        feats = F.normalize(feats, dim=-1)
        return feats.cpu().float().numpy()

    out_dim = 4 * (target_size // 8) * (target_size // 8)
    info = {"hf_id": hf_id, "output_dim": out_dim, "target_size": target_size}
    return embed, info, model


# ---------------------------------------------------------------------
# CORnet-S — bio-inspired V1→V2→V4→IT recurrent CNN (Brain-Score top)
# ---------------------------------------------------------------------

def load_cornet_s(device: str = "cuda"):
    """CORnet-S from DiCarlo lab — 4-stage recurrent CNN with V1/V2/V4/IT
    blocks matching primate ventral stream anatomy. We extract the
    pre-classifier features (after avgpool + flatten, before the 1000-class
    linear head) as a 512-d image embedding."""
    import cornet
    import torchvision.transforms as T
    model_dp = cornet.cornet_s(pretrained=True)  # DataParallel wrapper
    model = model_dp.module if hasattr(model_dp, "module") else model_dp
    model = model.to(device).eval()
    # CORnet expects 224×224 ImageNet-normalized input
    transform = T.Compose([
        T.Resize(256, antialias=True),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # Register hook on decoder.flatten so we get features pre-classifier
    feats_buf = {}
    def hook(_m, _inp, out):
        feats_buf["x"] = out.detach()
    h = model.decoder.flatten.register_forward_hook(hook)

    def embed(images: Sequence[Image.Image]) -> np.ndarray:
        batch = torch.stack([transform(img) for img in images]).to(device)
        with torch.no_grad():
            _ = model(batch)
        feats = feats_buf["x"]
        feats = F.normalize(feats, dim=-1)
        return feats.cpu().float().numpy()

    info = {"hf_id": "dicarlolab/CORnet-S", "output_dim": 512}
    return embed, info, model


# ---------------------------------------------------------------------
# Pixel baseline (control N03)
# ---------------------------------------------------------------------

def load_pixel_baseline(device: str = "cuda", target_size: int = 224):
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize(target_size, antialias=True),
        T.CenterCrop(target_size),
        T.ToTensor(),  # [0, 1]
    ])

    def embed(images: Sequence[Image.Image]) -> np.ndarray:
        batch = torch.stack([transform(img) for img in images])
        feats = batch.flatten(1)  # (B, 3*H*W)
        feats = F.normalize(feats, dim=-1)
        return feats.numpy()

    info = {"hf_id": None, "output_dim": 3 * target_size * target_size}
    return embed, info, None


# ---------------------------------------------------------------------
# Face-trained backbones — facenet-pytorch InceptionResnetV1
# ---------------------------------------------------------------------

def load_facenet_pytorch(pretrained: str, device: str = "cuda"):
    """facenet-pytorch InceptionResnetV1.
    pretrained ∈ {'vggface2', 'casia-webface'}. Expects [-1,1] 160×160 input;
    produces 512-dim L2-normalized face-identity embedding."""
    from facenet_pytorch import InceptionResnetV1
    import torchvision.transforms as T
    model = InceptionResnetV1(pretrained=pretrained).to(device).eval()
    transform = T.Compose([
        T.Resize(160, antialias=True),
        T.CenterCrop(160),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    def embed(images: Sequence[Image.Image]) -> np.ndarray:
        batch = torch.stack([transform(img) for img in images]).to(device)
        with torch.no_grad():
            feats = model(batch)
            feats = F.normalize(feats, dim=-1)
        return feats.cpu().float().numpy()

    info = {"hf_id": f"facenet-pytorch:InceptionResnetV1:{pretrained}",
            "output_dim": 512}
    return embed, info, model


# ---------------------------------------------------------------------
# Untrained ViT-B/16 (control N02)
# ---------------------------------------------------------------------

def load_arcface_onnx(onnx_path: str, device: str = "cuda"):
    """ArcFace ResNet-100 + additive angular margin loss (AuraFace checkpoint).

    Input expected: BGR-ordered 112x112 image, normalized to [-1, 1] via
    (x - 127.5) / 127.5. Outputs 512-d L2-normalized embedding.
    NPZ key for extracted embeddings: P23_arcface_auraface.
    """
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" \
        else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)

    def embed(images: Sequence[Image.Image]) -> np.ndarray:
        batch = []
        for img in images:
            arr = np.asarray(img.convert("RGB").resize((112, 112), Image.BILINEAR)).astype(np.float32)
            arr = arr[..., ::-1]  # RGB -> BGR
            arr = (arr - 127.5) / 127.5
            arr = arr.transpose(2, 0, 1)
            batch.append(arr)
        batch_np = np.stack(batch, axis=0)
        feats = sess.run(None, {"data": batch_np})[0]
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True).clip(1e-9)
        return feats.astype(np.float32)

    info = {"hf_id": "fal/AuraFace-v1 (glintr100.onnx)",
            "output_dim": 512,
            "preprocessing": "BGR 112x112 [-1,1]"}
    return embed, info, None  # ONNX session has no nn.Module


def load_untrained_vit(device: str = "cuda", dtype=torch.float16):
    import timm
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    model = model.to(device).to(dtype).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)

    def embed(images: Sequence[Image.Image]) -> np.ndarray:
        batch = torch.stack([transform(img) for img in images]).to(device).to(dtype)
        with torch.no_grad():
            feats = model(batch)
            feats = F.normalize(feats, dim=-1)
        return feats.cpu().float().numpy()

    info = {"hf_id": "vit_base_patch16_224 (untrained)",
            "output_dim": int(model.num_features)}
    return embed, info, model


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

REGISTRY: dict[str, Callable] = {
    # CLIP family
    "P02_clip_b32":    lambda: load_clip_hf("openai/clip-vit-base-patch32"),
    "P03_clip_l14":    lambda: load_clip_hf("openai/clip-vit-large-patch14"),
    "P04_clip_h14":    lambda: load_clip_hf("laion/CLIP-ViT-H-14-laion2B-s32B-b79K"),
    "P05_clip_g14":    lambda: load_clip_hf("laion/CLIP-ViT-g-14-laion2B-s12B-b42K"),
    "P06_clip_bigG14": lambda: load_clip_hf("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"),
    # DINOv2 (self-supervised; commonly considered more "object-shape" focused)
    "P07_dinov2_base":  lambda: load_dinov2_hf("facebook/dinov2-base"),
    "P08_dinov2_large": lambda: load_dinov2_hf("facebook/dinov2-large"),
    "P09_dinov2_giant": lambda: load_dinov2_hf("facebook/dinov2-giant"),
    # MAE (pixel-prediction self-supervised — "image-statistics" prior)
    "P10_mae_huge":     lambda: load_mae_hf("facebook/vit-mae-huge"),
    # VAE (true pixel-statistics encoder)
    "P11_sdxl_vae":     lambda: load_sdxl_vae(),
    # Face-trained backbones (Q003)
    "P17_facenet_vggface2":    lambda: load_facenet_pytorch("vggface2"),
    "P18_facenet_casiawebface": lambda: load_facenet_pytorch("casia-webface"),
    # Bio-inspired (Idea-003, Q010)
    "P22_cornet_s":            lambda: load_cornet_s(),
    # Modern face-recognition SOTA (Idea-003 sub-path (a) probe — E030)
    "P23_arcface_auraface":    lambda: load_arcface_onnx("/workspace/models/auraface/glintr100.onnx"),
    # Image-text contrastive variants (Q007) — discriminate CLIP-specific vs family-general
    "P19_siglip_base_384":      lambda: load_siglip_hf("google/siglip-base-patch16-384"),
    "P20_siglip_so400m":        lambda: load_siglip_hf("google/siglip-so400m-patch14-384"),
    "P21_metaclip_h14":         lambda: load_metaclip_hf("facebook/metaclip-h14-fullcc2.5b"),
    # Negative controls
    "N02_untrained_vit": lambda: load_untrained_vit(),
    "N03_pixel":         lambda: load_pixel_baseline(),
}


def free_model(model_obj):
    """Release a model's GPU memory; safe to call with None."""
    if model_obj is not None:
        try:
            model_obj.cpu()
        except Exception:
            pass
        del model_obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
