import easyocr

class OCRExtractor:
    def __init__(self, languages=['en'], gpu=True):
        print("Loading EasyOCR...")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print("EasyOCR ready!")

    def extract(self, imagePath: str) -> list:
        """
        Returns a list of detections, each with:
        - bbox: 4 corner points [[x,y], [x,y], [x,y], [x,y]]
        - text: detected string
        - confidence: float 0-1
        """
        results = self.reader.readtext(imagePath)
        detections = []
        for bbox, text, confidence in results:
            clean_bbox = [[int(x), int(y)] for x, y in bbox]
            detections.append({
                "bbox": clean_bbox,
                "text": text,
                "confidence": round(float(confidence), 3)
            })
        return {'OCR Result':detections}


if __name__ == "__main__":
    extractor = OCRExtractor()
    results = extractor.extract("electricity.jpg")
    for det in results:
        print(f"Text: {det['text']!r}  |  Confidence: {det['confidence']}  |  BBox: {det['bbox']}")