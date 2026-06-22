from PIL import Image

def compute_text_coverage(image_path: str, ocr_detections: list) -> float:
    """
    Returns the fraction of image area covered by OCR-detected text boxes.
    """
    image = Image.open(image_path)
    img_area = image.width * image.height
    if img_area == 0:
        return 0.0

    text_area = 0
    for det in ocr_detections:
        bbox = det["bbox"]
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        text_area += w * h

    return text_area / img_area


def classify_image_type(image_path: str, ocr_detections: list, threshold: float = 0.03) -> str:
    """
    Returns 'document' if text coverage exceeds threshold, else 'photo'.
    threshold=0.03 means: if >3% of image area is text, treat as document/diagram.
    """
    coverage = compute_text_coverage(image_path, ocr_detections)
    return "document" if coverage >= threshold else "photo"
