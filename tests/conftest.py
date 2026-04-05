"""Pytest configuration — mock unavailable C extensions before any imports."""
import sys
from unittest.mock import MagicMock

# audioop was removed in Python 3.13; pydub (pulled by gradio) depends on it.
# Mock it so tests can import app.main without a broken gradio install.
for _mod in ("audioop", "pyaudioop"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
