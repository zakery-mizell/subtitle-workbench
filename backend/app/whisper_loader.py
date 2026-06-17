from __future__ import annotations

import importlib
import sys
from pathlib import Path

from .config import settings


def import_whisper():
    source_dir = Path(settings.whisper_source_dir)
    if source_dir.exists():
        candidate = str(source_dir.resolve())
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    return importlib.import_module("whisper")
