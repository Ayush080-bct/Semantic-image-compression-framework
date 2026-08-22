"""Streamlit entry point for the modular dashboard."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from Dashboard.app import *  # noqa: F401, F403, E402