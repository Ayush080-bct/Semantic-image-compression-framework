"""Streamlit dashboard for inspecting semantic image representations."""

import io
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dashboard.processing import process_upload
from Dashboard.representation import draw_text_boxes, parse_result, token_count


st.set_page_config(page_title="Semantic Image Compression", page_icon=":bar_chart:", layout="wide")


def format_bytes(value: int) -> str:
    return f"{value:,}"


def build_measurement_rows(result: dict, xml_text: str, parsed: dict) -> list[dict[str, str]]:
    xml_tokens = token_count(xml_text)
    image_base64_tokens = token_count(result["image_bytes"].hex())
    xml_size = len(xml_text.encode("utf-8"))
    image_size = max(1, len(result["image_bytes"]))
    return [
        {"Metric": "XML / image bytes", "Value": f"{xml_size / image_size:.4f}", "Interpretation": "Storage-size proxy"},
        {"Metric": "XML character count", "Value": format_bytes(len(xml_text)), "Interpretation": "Representation-size measure"},
        {"Metric": "XML tokens", "Value": str(xml_tokens) if xml_tokens is not None else "Unavailable", "Interpretation": "cl100k_base estimate"},
        {"Metric": "Image payload tokens", "Value": str(image_base64_tokens) if image_base64_tokens is not None else "Unavailable", "Interpretation": "Hex payload proxy, not multimodal billing"},
        {"Metric": "Dense regions", "Value": str(len(parsed["region_nodes"])), "Interpretation": "Florence region count"},
    ]


def render_result(result: dict) -> None:
    parsed = result["parsed"]
    image = Image.open(io.BytesIO(result["image_bytes"])).convert("RGB")
    xml_text = result["xml"]

    st.divider()
    st.subheader(result["image_name"])
    metric_columns = st.columns(6)
    metric_columns[0].metric("Route", parsed["image_type"].title())
    metric_columns[1].metric("OCR regions", len(parsed["text_nodes"]))
    metric_columns[2].metric("Mean OCR confidence", f"{parsed['mean_confidence']:.3f}")
    metric_columns[3].metric("Processing time", f"{result['elapsed']:.2f} s")
    metric_columns[4].metric("Image bytes", format_bytes(len(result["image_bytes"])))
    metric_columns[5].metric("XML bytes", format_bytes(len(xml_text.encode("utf-8"))))

    left, right = st.columns(2)
    with left:
        st.image(image, caption="Input image", use_container_width=True)
    with right:
        st.image(draw_text_boxes(image, parsed["text_nodes"]), caption="OCR regions", use_container_width=True)

    st.subheader("Representation")
    st.write(parsed["caption"] or "No caption returned.")
    st.code(xml_text, language="xml")
    st.download_button("Download XML", xml_text, file_name=f"{Path(result['image_name']).stem}.xml", mime="application/xml")

    st.subheader("Observed measurements")
    st.table(build_measurement_rows(result, xml_text, parsed))

    if parsed["text_nodes"]:
        st.subheader("OCR extraction")
        st.dataframe(
            [{"Text": node.text or "", "Confidence": node.attrib.get("confidence", ""), "Bounding box": node.attrib.get("bbox", "")} for node in parsed["text_nodes"]],
            use_container_width=True,
            hide_index=True,
        )


st.title("Semantic Image Compression")
st.caption("Inspect the extracted representation and collect reproducible Chapter 4 measurements.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
run_pipeline = st.button("Run semantic extraction", type="primary", disabled=uploaded_file is None)

if run_pipeline and uploaded_file is not None:
    try:
        result = process_upload(uploaded_file)
        result["parsed"] = parse_result(result["xml"])
        st.session_state["result"] = result
    except Exception as error:
        st.error(f"Extraction failed: {error}")
        st.stop()

result = st.session_state.get("result")
if result:
    render_result(result)

st.sidebar.header("Experiment notes")
st.sidebar.write("The size and token values are representation proxies. They do not establish semantic fidelity or LLM billing savings by themselves.")
st.sidebar.write("Document routing threshold: 3% OCR box coverage")
st.sidebar.write("Photo OCR retention threshold: 0.80 confidence")