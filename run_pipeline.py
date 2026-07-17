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
        self.imagePreprocessor =ImagePreprocessor()
        self.semanticMapper = SemanticMapper()

    def run(self, image_path):
        ocr_result = self.ocr.extract(image_path)
        vlm_result = self.florence.extract(image_path)
        image_type = self.imagePreprocessor.classify_image_type(image_path, ocr_result['OCR Result'])

        result = {
            "image_type": image_type,
            "VLM": vlm_result,
            "OCR": ocr_result,
        }

        return self.semanticMapper.build_xml(result)

if __name__ == '__main__':
    if len(sys.argv) != 2: 
        print('Provide a image')
        sys.exit(1)

    print(PipelineManager().run(sys.argv[1]))