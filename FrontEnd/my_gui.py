from PySide6.QtWidgets import QApplication , QMainWindow , QLabel , QVBoxLayout,QHBoxLayout , QWidget , QPushButton,QFileDialog,QScrollArea,QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap


class MainWindow:

    def __init__(self , executeCallback):

        self.executeCallback = executeCallback

        self.mainWindow = QMainWindow()
        self.mainWindow.setWindowTitle('Title')
        self.mainContainer = QWidget()
        self.mainWindow.setCentralWidget(self.mainContainer)
        self.mainWindow.resize(1024, 768) 

        self.mainLayout = QHBoxLayout(self.mainContainer)

        self.controlContainer = QWidget()
        self.controlLayout = QVBoxLayout(self.controlContainer)

        self.imageContainer = QLabel()
        self.imageContainer.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.imageContainer.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.imageContainer.setMinimumSize(100, 100)

        self.executeButton = QPushButton('Execute')
        self.executeButton.clicked.connect(self._execute)

        self.imagePickButton = QPushButton('Pick Image',self.controlContainer)
        self.imagePickButton.clicked.connect(self.pick_image)

        self.outputArea = QScrollArea()
        self.outputArea.setWidgetResizable(True)

        self.outputContainer = QLabel()
        self.outputContainer.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.outputArea.setWidget(self.outputContainer)


        self.controlLayout.addWidget(self.executeButton)
        self.controlLayout.addWidget(self.imagePickButton)
        self.controlLayout.addWidget(self.outputArea)

        self.mainLayout.addWidget(self.controlContainer,1)
        self.mainLayout.addWidget(self.imageContainer,1)


        self.mainWindow.resizeEvent = self.on_window_resize

    def pick_image(self):
        
        self.fileName,_ = QFileDialog.getOpenFileName(self.mainWindow,
            self.mainWindow.tr("Open Image"), "/home/jana", self.mainWindow.tr("Image Files (*.png *.jpg *.bmp *.jpeg)"))
        
        if self.fileName:
            self.master_pixmap = QPixmap(self.fileName)
            self.scale_and_set_image()
    
    def _execute(self):
        if self.fileName:
            result = self.executeCallback(self.fileName)
            self.outputContainer.setText(result)
            
        else:
            print('Select a image')

    def scale_and_set_image(self):
        if hasattr(self, 'master_pixmap') and not self.master_pixmap.isNull():

            target_size = self.imageContainer.size()

            true_width = self.master_pixmap.width()
            true_height = self.master_pixmap.height()

            bounded_width = min(target_size.width(), true_width)
            bounded_height = min(target_size.height(), true_height)

            # 4. Scale cleanly keeping original proportions intact
            scaled_pixmap = self.master_pixmap.scaled(
                bounded_width,bounded_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation 
            )

            self.imageContainer.setPixmap(scaled_pixmap)
    
    def on_window_resize(self, event):
        if event:
            event.accept()
        # Dynamically recalculate scale parameters when window changes size
        self.scale_and_set_image()

