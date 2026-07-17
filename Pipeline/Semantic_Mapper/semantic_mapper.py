import xml.etree.ElementTree as ET
from xml.dom import minidom

class SemanticMapper:
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

    def build_xml(self,extraction_result: dict, ocr_confidence_threshold: float = 0.8) -> str:
        """
        Takes the merged Florence-2 + EasyOCR extraction result and
        builds a compact XML document.
        """

        root = ET.Element("ImageSemantic", type=extraction_result["image_type"])

        vlmEl = ET.SubElement(root, "VLM")
        vlmCaptionEl = ET.SubElement(vlmEl, "Caption")
        vlmCaptionEl.text = (
            extraction_result["VLM"]
            .get("VLM Result", {})
            .get("caption", {})
            .get("<DETAILED_CAPTION>", "")
        )

        vlmObjectsEl = ET.SubElement(vlmEl, "Objects")

        objectDetections = (
            extraction_result["VLM"]
            .get("VLM Result", {})
            .get("objects", {})
            .get("<OD>", {})
            .get("labels", [])
        )
        objectBoxes = (
            extraction_result["VLM"]
            .get("VLM Result", {})
            .get("objects", {})
            .get("<OD>", {})
            .get("bboxes", [])
        )

        for index, obj in enumerate(objectDetections):
            objEl = ET.SubElement(
                vlmObjectsEl, "object", bbox=self.format_bbox(objectBoxes[index])
            )
            objEl.text = obj

        vlmDenseRegionsEl = ET.SubElement(vlmEl, "DenseRegions")

        denseRegionDetections = (
            extraction_result["VLM"]
            .get("VLM Result", {})
            .get("dense_regions", {})
            .get("<DENSE_REGION_CAPTION>", {})
            .get("labels", [])
        )
        denseRegionBoxes = (
            extraction_result["VLM"]
            .get("VLM Result", {})
            .get("dense_regions", {})
            .get("<DENSE_REGION_CAPTION>", {})
            .get("bboxes", [])
        )

        for index, region in enumerate(denseRegionDetections):
            regionEl = ET.SubElement(
                vlmDenseRegionsEl, "DenseRegion", bbox=self.format_bbox(denseRegionBoxes[index])
            )
            regionEl.text = region

        ocrTextRegionsEl = ET.SubElement(root, "OCR")
        for det in extraction_result.get("OCR", []).get("OCR Result"):
            text_el = ET.SubElement(
                ocrTextRegionsEl,
                "text",
                bbox=self.format_bbox(det["bbox"]),
                confidence=str(det["confidence"]),
            )
            text_el.text = det["text"]

        # Include high-confidence OCR hits even for photos

        highConfText = [
            det
            for det in extraction_result.get("OCR", {}).get("OCR Result", [])
            if det["confidence"] >= ocr_confidence_threshold
        ]
        if highConfText:
            textRegionsEl = ET.SubElement(root, "textRegions")
            for det in highConfText:
                textEl = ET.SubElement(
                    textRegionsEl,
                    "text",
                    bbox=self.format_bbox(det["bbox"]),
                    confidence=str(det["confidence"]),
                )
                textEl.text = det["text"]

        # Pretty-print
        roughBytes = ET.tostring(root, encoding="utf-8", xml_declaration=False)
        reparsed = minidom.parseString(roughBytes)
        return reparsed.toprettyxml(indent="  ").replace('<?xml version="1.0" ?>\n', "")


if __name__ == "__main__":
    from ..Vision_Language_Model.florence_extractor import FlorenceExtractor
    from ..OCR_Model.easyocr_extractor import OCRExtractor
    from ..Image_Preprocessor.image_preprocessor import ImagePreprocessor
    from PIL import Image

    florence = FlorenceExtractor()
    ocr = OCRExtractor()
    imgPreprocessor=ImagePreprocessor()
    semanticMapper=SemanticMapper()

    image_path = "image.png"  # change to test photo.jpg too
    ocr_results = ocr.extract(image_path)
    image_type = imgPreprocessor.classify_image_type(image_path, ocr_results)
    image = Image.open(image_path).convert("RGB")

    result = {
        "image_type": image_type,
        "caption": florence.run_task(image, "<DETAILED_CAPTION>"),
        "ocr": ocr_results,
    }

    if image_type == "photo":
        result["objects"] = florence.run_task(image, "<OD>")
        result["dense_regions"] = florence.run_task(image, "<DENSE_REGION_CAPTION>")

    xml_output = semanticMapper.build_xml(result)
    print(xml_output)
