import sys  
from PIL import Image

class PipelineManager:
    def __init__(self):

        from Pipeline.Vision_Language_Model.florence_extractor import FlorenceExtractor
        from Pipeline.OCR_Model.easyocr_extractor import OCRExtractor
        from Pipeline.Image_Preprocessor.router import classify_image_type 
        from Pipeline.Semantic_Mapper.semantic_mapper import build_xml
      
        self.florence = FlorenceExtractor()
        self.ocr = OCRExtractor()
        self.classify_image_type = classify_image_type
        self.build_xml = build_xml

    def run(self, image_path):
        ocr_result = self.ocr.extract(image_path)
        vlm_result = self.florence.extract(image_path)
        image_type = self.classify_image_type(image_path, ocr_result['OCR Result'])

        result = {
            "image_type": image_type,
            "VLM": vlm_result,
            "OCR": ocr_result,
        }

        return self.build_xml(result)

if __name__ == '__main__':
    if len(sys.argv) != 2: 
        print('Provide a image')
        sys.exit(1)

    print(PipelineManager().run(sys.argv[1]))