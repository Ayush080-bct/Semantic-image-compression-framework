"""Pipeline lifecycle and uploaded-image processing."""

import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Pipeline.pipeline import SemanticImagePipeline  # noqa: E402


@st.cache_resource(show_spinner="Loading EasyOCR and Florence-2...")
def get_pipeline() -> SemanticImagePipeline:
    return SemanticImagePipeline()


def process_upload(uploaded_file) -> dict:
    image_bytes = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix) as input_file:
        input_file.write(image_bytes)
        input_file.flush()
        started = time.perf_counter()
        xml_text = get_pipeline().process(input_file.name)

    return {
        "image_bytes": image_bytes,
        "image_name": uploaded_file.name,
        "xml": xml_text,
        "elapsed": time.perf_counter() - started,
    }