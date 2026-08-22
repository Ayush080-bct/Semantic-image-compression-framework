"""
Semantic Image Compression Pipeline
-------------------------------------
Converts an image into a compact XML semantic representation
using Florence-2 (vision) + EasyOCR (text), routed by content type.
"""

import argparse
from PIL import Image
from Pipeline.Image_Preprocessor.image_preprocessor import ImagePreprocessor
from Pipeline.OCR_Model.easyocr_extractor import OCRExtractor
from Pipeline.Vision_Language_Model.florence_extractor import FlorenceExtractor
from Pipeline.Semantic_Mapper.semantic_mapper import SemanticMapper


class SemanticImagePipeline:
    """
    Wraps Florence-2 + EasyOCR + routing + XML mapping into a single
    reusable interface. Models are loaded once at init for efficiency
    when processing multiple images.
    """

    def __init__(self, ocr_confidence_threshold: float = 0.8, document_threshold: float = 0.03):
        self.florence = FlorenceExtractor()
        self.ocr = OCRExtractor()
        self.preprocessor = ImagePreprocessor()
        self.mapper = SemanticMapper()
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.document_threshold = document_threshold

    def process(self, image_path: str) -> str:
        """
        Runs the full pipeline on a single image and returns the XML string.
        """
        ocr_results = self.ocr.extract(image_path)
        image_type = self.preprocessor.classify_image_type(
            image_path, ocr_results["OCR Result"], threshold=self.document_threshold
        )
        image = Image.open(image_path).convert("RGB")

        result = {
            "image_type": image_type,
            "caption": self.florence.run_task(image, "<DETAILED_CAPTION>"),
            "ocr": ocr_results,
        }

        if image_type == "photo":
            result["dense_regions"] = self.florence.run_task(image, "<DENSE_REGION_CAPTION>")

        return self.mapper.build_xml(
            result, ocr_confidence_threshold=self.ocr_confidence_threshold
        )


def main():
    parser = argparse.ArgumentParser(description="Semantic Image Compression Pipeline")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("-o", "--output", help="Optional output file path for the XML")
    args = parser.parse_args()

    pipeline = SemanticImagePipeline()
    xml_output = pipeline.process(args.image_path)

    if args.output:
        with open(args.output, "w") as f:
            f.write(xml_output)
        print(f"XML written to {args.output}")
    else:
        print(xml_output)


if __name__ == "__main__":
    main()