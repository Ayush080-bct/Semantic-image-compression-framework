import sys
from PIL import Image

class PipelineManager:
    def __init__(self):
        from Pipeline.Vision_Language_Model.florence_extractor import FlorenceExtractor
        from Pipeline.OCR_Model.easyocr_extractor import OCRExtractor
        from Pipeline.Image_Preprocessor.image_preprocessor import ImagePreprocessor
        from Pipeline.Semantic_Mapper.semantic_mapper import SemanticMapper

        self.florence = FlorenceExtractor()
        self.ocr = OCRExtractor()
        self.imagePreprocessor = ImagePreprocessor()
        self.semanticMapper = SemanticMapper()

    def run(self, image_path):
        # Step 1: Always run OCR (needed for routing + text extraction)
        ocr_result = self.ocr.extract(image_path)
        detections = ocr_result['OCR Result']

        # Step 2: Classify image type using OCR coverage
        image_type = self.imagePreprocessor.classify_image_type(image_path, detections)

        # Step 3: Run Florence-2 tasks based on image type
        image = Image.open(image_path).convert("RGB")

        if image_type == "document":
            vlm_result = {
                'VLM Result': {
                    "caption": self.florence.run_task(image, "<DETAILED_CAPTION>"),
                }
            }
        else:  # photo
            vlm_result = {
                'VLM Result': {
                    "caption": self.florence.run_task(image, "<DETAILED_CAPTION>"),
                    "dense_regions": self.florence.run_task(image, "<DENSE_REGION_CAPTION>"),
                }
            }

        # Step 4: Build XML
        result = {
            "image_type": image_type,
            "VLM": vlm_result,
            "OCR": ocr_result,
        }

        return self.semanticMapper.build_xml(result)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Provide an image')
        sys.exit(1)

    print(PipelineManager().run(sys.argv[1]))