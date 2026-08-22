"""Streamlit entry point for the modular dashboard."""

import sys
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "Dashboard" / "app.py"), run_name="__main__")