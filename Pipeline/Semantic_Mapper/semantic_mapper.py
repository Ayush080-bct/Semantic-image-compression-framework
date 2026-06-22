import xml.etree.ElementTree as ET
from xml.dom import minidom


def format_bbox(bbox) -> str:
    """
    Normalizes different bbox formats into 'x1,y1,x2,y2' string.
    Handles both Florence-2 style [x1,y1,x2,y2] and EasyOCR style
    [[x,y],[x,y],[x,y],[x,y]] (4 corner points).
    """
    if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
        x1, y1, x2, y2 = bbox
    else:
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

    return f"{round(x1)},{round(y1)},{round(x2)},{round(y2)}"


def build_xml(extraction_result: dict, ocr_confidence_threshold: float = 0.8) -> str:
    """
    Takes the merged Florence-2 + EasyOCR extraction result and
    builds a compact XML document.
    """
    root = ET.Element("image_semantic", type=extraction_result["image_type"])

    caption_text = extraction_result["caption"].get("<DETAILED_CAPTION>", "")
    caption_el = ET.SubElement(root, "caption")
    caption_el.text = caption_text

    if extraction_result["image_type"] == "document":
        text_regions = ET.SubElement(root, "text_regions")
        for det in extraction_result.get("ocr", []):
            text_el = ET.SubElement(
                text_regions, "text",
                bbox=format_bbox(det["bbox"]),
                confidence=str(det["confidence"])
            )
            text_el.text = det["text"]

    else:  # photo
        dense_data = extraction_result.get("dense_regions", {}).get("<DENSE_REGION_CAPTION>", {})
        dense_bboxes = dense_data.get("bboxes", [])
        dense_labels = dense_data.get("labels", [])

        if dense_bboxes:
            regions_el = ET.SubElement(root, "regions")
            for bbox, label in zip(dense_bboxes, dense_labels):
                region_el = ET.SubElement(
                    regions_el, "region",
                    bbox=format_bbox(bbox)
                )
                region_el.text = label

        # Include high-confidence OCR hits even for photos
        high_conf_text = [
        det for det in extraction_result.get("ocr", [])
        if det["confidence"] >= ocr_confidence_threshold
        ]
        if high_conf_text:
            text_regions = ET.SubElement(root, "text_regions")
            for det in high_conf_text:
                text_el = ET.SubElement(
                    text_regions, "text",
                    bbox=format_bbox(det["bbox"]),
                    confidence=str(det["confidence"])
                )
                text_el.text = det["text"]

    # Pretty-print
    rough_string = ET.tostring(root, encoding="unicode")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ").replace('<?xml version="1.0" ?>\n', '')


if __name__ == "__main__":
    from florence_extractor import FlorenceExtractor
    from easyocr_extractor import OCRExtractor
    from router import classify_image_type
    from PIL import Image

    florence = FlorenceExtractor()
    ocr = OCRExtractor()

    image_path = "image.png"  # change to test photo.jpg too
    ocr_results = ocr.extract(image_path)
    image_type = classify_image_type(image_path, ocr_results)
    image = Image.open(image_path).convert("RGB")

    result = {
        "image_type": image_type,
        "caption": florence.run_task(image, "<DETAILED_CAPTION>"),
        "ocr": ocr_results,
    }

    if image_type == "photo":
        result["objects"] = florence.run_task(image, "<OD>")
        result["dense_regions"] = florence.run_task(image, "<DENSE_REGION_CAPTION>")

    xml_output = build_xml(result)
    print(xml_output)