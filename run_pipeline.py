import sys  

if len(sys.argv) != 2: 
    print('Provide a image')
    sys.exit(1)


from Pipeline.Vision_Language_Model.florence_extractor import FlorenceExtractor
from Pipeline.OCR_Model.easyocr_extractor import OCRExtractor
from Pipeline.Image_Preprocessor.router import classify_image_type 
from Pipeline.Semantic_Mapper.semantic_mapper import build_xml
from PIL import Image


def run_pipeline():
    florence = FlorenceExtractor()
    ocr = OCRExtractor()

    image_path = sys.argv[1]
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

if __name__ == '__main__':
    run_pipeline()