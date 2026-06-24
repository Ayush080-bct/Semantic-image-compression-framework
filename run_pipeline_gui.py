from tkinter import *
from tkinter import ttk
from FrontEnd.my_gui import MyGui
from PIL import Image, ImageTk



def runPipeline(imagePath):

    from Pipeline.Vision_Language_Model.florence_extractor import FlorenceExtractor
    from Pipeline.OCR_Model.easyocr_extractor import OCRExtractor
    from Pipeline.Image_Preprocessor.router import classify_image_type 
    from Pipeline.Semantic_Mapper.semantic_mapper import build_xml
    from PIL import Image

    florence = FlorenceExtractor()
    ocr = OCRExtractor()

    image_path = imagePath  # change to test photo.jpg too
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
    return xml_output

def handle_print():
    
    imgOpen = Image.open(gui.imagePath.get())
    imgOpen.thumbnail((350, 350))
    image = ImageTk.PhotoImage(imgOpen)
    
    lbl = ttk.Label(gui.imageFrame,image=image)
    lbl.image = image
    lbl.grid()

    gui.root.update()

    result = runPipeline(gui.imagePath.get())
    return result


gui = MyGui(Tk(),handle_print)




gui.root.mainloop()