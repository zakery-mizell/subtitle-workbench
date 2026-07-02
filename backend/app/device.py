from __future__ import annotations


def select_device(preference: str = "auto") -> str:
    if preference in ("cuda", "mps", "cpu"):
        return preference

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
