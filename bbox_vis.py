from PIL import Image, ImageDraw, ImageFont
import json

def visualize_ocr(image_path, detections):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for det in detections:
        bbox = det["bbox"]
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        color = "green" if det["confidence"] >= 0.8 else "red"
        draw.rectangle([min(xs), min(ys), max(xs), max(ys)], outline=color, width=2)
        draw.text((min(xs), min(ys)-12), f"{det['text']} ({det['confidence']:.2f})", fill=color)
    img.save("ocr_visualization.png")

from Pipeline.OCR_Model.easyocr_extractor import OCRExtractor
ocr = OCRExtractor()
result = ocr.extract("image.png")
visualize_ocr("image.png", result["OCR Result"])
print("Saved ocr_visualization.png")