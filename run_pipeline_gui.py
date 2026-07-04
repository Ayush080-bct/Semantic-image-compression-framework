from PySide6.QtWidgets import QApplication
from FrontEnd.my_gui import MainWindow
from Pipeline.Vision_Language_Model.florence_extractor import FlorenceExtractor
from Pipeline.OCR_Model.easyocr_extractor import OCRExtractor
from Pipeline.Image_Preprocessor.router import classify_image_type 
from Pipeline.Semantic_Mapper.semantic_mapper import build_xml
from PIL import Image
from run_pipeline import PipelineManager

app = QApplication()

window = MainWindow(PipelineManager().run)
window.mainWindow.show()
app.exec()