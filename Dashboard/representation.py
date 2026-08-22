"""Helpers for inspecting and displaying semantic representations."""

import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw


def confidence_values(text_nodes: list) -> list[float]:
    values = []
    for node in text_nodes:
        try:
            values.append(float(node.attrib["confidence"]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def parse_bbox(node) -> tuple[int, int, int, int] | None:
    try:
        return tuple(int(value) for value in node.attrib["bbox"].split(","))
    except (KeyError, ValueError):
        return None


def token_count(value: str) -> int | None:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(value))
    except Exception:
        return None


def parse_result(xml_text: str) -> dict:
    root = ET.fromstring(xml_text)
    text_nodes = root.findall("./text_regions/text")
    region_nodes = root.findall("./regions/region")
    confidences = confidence_values(text_nodes)
    return {
        "image_type": root.attrib.get("type", "unknown"),
        "caption": root.findtext("caption", default=""),
        "text_nodes": text_nodes,
        "region_nodes": region_nodes,
        "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
    }


def draw_text_boxes(image: Image.Image, text_nodes: list) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for node in text_nodes:
        bbox = parse_bbox(node)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        draw.rectangle((x1, y1, x2, y2), outline="#e4572e", width=3)
        draw.text((x1, max(0, y1 - 16)), node.text or "", fill="#e4572e")
    return annotated