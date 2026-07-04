import sys  
from PIL import Image

class PipelineManager:
    def __init__(self):

        from Pipeline.Vision_Language_Model.florence_extractor import FlorenceExtractor
        from Pipeline.OCR_Model.easyocr_extractor import OCRExtractor
        from Pipeline.Image_Preprocessor.router import classify_image_type 
        from Pipeline.Semantic_Mapper.semantic_mapper import build_xml
      


        # Models are loaded into memory exactly once here
        self.florence = FlorenceExtractor()
        self.ocr = OCRExtractor()
        self.classify_image_type = classify_image_type
        self.build_xml = build_xml

    def run(self, image_path):
        ocr_results = self.ocr.extract(image_path)
        image_type = self.classify_image_type(image_path, ocr_results)
        image = Image.open(image_path).convert("RGB")

        result = {
            "image_type": image_type,
            "caption": self.florence.run_task(image, "<DETAILED_CAPTION>"),
            "ocr": ocr_results,
        }

        if image_type == "photo":
            result["objects"] = self.florence.run_task(image, "<OD>")
            result["dense_regions"] = self.florence.run_task(image, "<DENSE_REGION_CAPTION>")

        return self.build_xml(result)

if __name__ == '__main__':
    if len(sys.argv) != 2: 
        print('Provide a image')
        sys.exit(1)

    print(PipelineManager().run(sys.argv[1]))