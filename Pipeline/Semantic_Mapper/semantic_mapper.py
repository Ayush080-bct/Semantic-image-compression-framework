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

    print(extraction_result["VLM"].get("VLM Result", {}))

    VLM_el = ET.SubElement(root, "VLM")
    VLMCaption_el = ET.SubElement(VLM_el,'Caption')
    VLMCaption_el.text = extraction_result["VLM"].get("VLM Result", {}).get('caption',{}).get('<DETAILED_CAPTION>','')

    VLMObjects_el = ET.SubElement(VLM_el,'Objects')
    VLMObjectDetection_el = ET.SubElement(VLMObjects_el,'ObjectDetection')

    objectDetections = extraction_result["VLM"].get("VLM Result", {}).get('objects',{}).get('<OD>',{}).get('labels',[])
    objectBoxes = extraction_result["VLM"].get("VLM Result", {}).get('objects',{}).get('<OD>',{}).get('bboxes',[])
    
    for index,obj in enumerate(objectDetections):
        obj_el = ET.SubElement(VLMObjectDetection_el , 'object',bbox = format_bbox(objectBoxes[index]))
        obj_el.text = obj


    text_regions = ET.SubElement(root, "text_regions")
    for det in extraction_result.get("OCR", []).get('OCR Result'):
        text_el = ET.SubElement(
            text_regions, "text",
            bbox=format_bbox(det["bbox"]),
            confidence=str(det["confidence"])
        )
        text_el.text = det["text"]

    
    dense_data = extraction_result.get("VLM", {}).get('VLM Result',{}).get("dense_regions", {}).get('<DENSE_REGION_CAPTION>',{})

    #print(dense_data)

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
    det for det in extraction_result.get("OCR", {}).get('OCR Result',[])
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
    rough_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=False)
    reparsed = minidom.parseString(rough_bytes)
    return  reparsed.toprettyxml(indent="  ").replace('<?xml version="1.0" ?>\n', '')


if __name__ == "__main__":
    from ..Vision_Language_Model.florence_extractor import FlorenceExtractor
    from ..OCR_Model.easyocr_extractor import OCRExtractor
    from ..Image_Preprocessor.router import classify_image_type
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