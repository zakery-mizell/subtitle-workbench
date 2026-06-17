from __future__ import annotations

import warnings
from contextlib import contextmanager
from collections.abc import Iterator


@contextmanager
def suppress_known_audio_stack_warnings() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"pyannote\.audio\.core\.io",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"speechbrain\.utils\.torch_audio_backend",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"Module 'speechbrain\.pretrained' was deprecated.*",
        )
        warnings.filterwarnings(
            "ignore",
            module=r"pyannote\.audio\.utils\.reproducibility",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"pyannote\.audio\.models\.blocks\.pooling",
        )
        yield
