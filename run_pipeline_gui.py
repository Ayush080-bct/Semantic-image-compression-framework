from PySide6.QtWidgets import QApplication
from FrontEnd.my_gui import MainWindow
from run_pipeline import PipelineManager

app = QApplication()

window = MainWindow(PipelineManager().run)
window.mainWindow.show()
app.exec()