from PySide6.QtWidgets import (
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QFileDialog,
    QScrollArea,
    QSizePolicy,
    QStyle,
)
from PySide6.QtCore import Qt, QTimer, QThreadPool, QRunnable, Slot, QObject, Signal
import re


class WorkerSignal(QObject):
    finished = Signal(int)
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(tuple)
    complete = Signal(tuple)


class Worker(QRunnable):

    def __init__(self):
        super().__init__()

        self.signals = WorkerSignal()

    def submit_callback(self, callback, *args, **kwargs):
        self.callback = callback
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self):
        result = self.callback(self.signals, *self.args, **self.kwargs)
        self.signals.result.emit(result)
        self.signals.finished.emit(0)


class MainWindow:

    def __init__(self, executeCallback):

        self.executeCallback = executeCallback
        self.threadPool = QThreadPool()
        self.fileNames = []
        self.masterPixmaps = []

        self.mainWindow = QMainWindow()
        self.mainWindow.setWindowTitle("Title")
        self.mainContainer = QWidget()
        self.mainWindow.setCentralWidget(self.mainContainer)
        self.mainWindow.resize(1024, 768)
        self.mainLayout = QHBoxLayout(self.mainContainer)

        self.controlContainer = QWidget()
        self.controlLayout = QVBoxLayout(self.controlContainer)

        self.imageContainers = []
        self.imageContainer = QScrollArea()
        self.imageContainer.setWidgetResizable(True)
        self.imageWidget = QWidget()
        self.imageLayout = QVBoxLayout(self.imageWidget)
        self.imageLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.imageContainer.setWidget(self.imageWidget)

        self.executeButton = QPushButton("Execute")
        self.executeButton.clicked.connect(self._execute)

        self.imagePickButton = QPushButton("Pick Image", self.controlContainer)
        self.imagePickButton.clicked.connect(self.pick_image)

        self.statusIndicator = QLabel(text="Not Running")

        self.outputArea = QScrollArea()
        self.outputArea.setWidgetResizable(True)
        self.outputContainer = QLabel(text='')
        self.outputContainer.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.outputContainer.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.outputArea.setWidget(self.outputContainer)

        self.controlLayout.addWidget(self.executeButton)
        self.controlLayout.addWidget(self.imagePickButton)
        self.controlLayout.addWidget(self.outputArea)
        self.controlLayout.addWidget(self.statusIndicator)

        self.mainLayout.addWidget(self.controlContainer, 1)
        self.mainLayout.addWidget(self.imageContainer, 1)

    def pick_image(self):
        fileNames, _ = QFileDialog.getOpenFileNames(
            self.mainWindow,
            self.mainWindow.tr("Open Image"),
            "/home/jana",
            self.mainWindow.tr("Image Files (*.png *.jpg *.bmp *.jpeg)"),
        )
        for name in fileNames:
            
            self.fileNames.append(name)

            filteredName = re.search(r"[^/]+$", name)

            imgContainerCard = QWidget()
            imgContainerCardLayout = QHBoxLayout(imgContainerCard)
            imgContainerCardLayout.setContentsMargins(0, 0, 0, 0)
            imgContainer = QLabel(filteredName.group(0))
            imgContainer.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
            )
            imgContainerStatus = QPushButton()
            imgContainerStatus.setFlat(True)
            imgContainerCardLayout.addWidget(imgContainer, 1)
            imgContainerCardLayout.addWidget(imgContainerStatus, 0)

            self.imageContainers.append(imgContainerCard)

        for imageContainer in self.imageContainers:
            self.imageLayout.addWidget(imageContainer, 0)

    def process_images(self, signals):
        output = []
        if len(self.fileNames) != 0:

            for index, fileName in enumerate(self.fileNames):
                signals.progress.emit((index, None))
                result = self.executeCallback(fileName)
                signals.complete.emit((index, None))
                output.append(result)

        else:
            print("Select a image")

        return output

    def _execute(self):

        self.outputContainer.setText('')

        self.statusIndicator.setText("Running")

        for  index ,names in enumerate(self.fileNames):
            self._update_img_container_status((index,None) , self.mainWindow.style().standardIcon(QStyle.SP_BrowserReload))

        workerThread = Worker()
        workerThread.signals.result.connect(
            lambda result: self.outputContainer.setText(" ".join(result))
        )
        workerThread.signals.progress.connect(
            lambda result: self._update_img_container_status(
                result,
                self.mainWindow.style().standardIcon(QStyle.SP_MediaPlay),
            )
        )
        workerThread.signals.complete.connect(
            lambda result: self._update_img_container_status(
                result,
                self.mainWindow.style().standardIcon(QStyle.SP_DialogApplyButton),
            )
        )
        workerThread.signals.finished.connect(
            lambda result: self.statusIndicator.setText("Finished")
        )
        workerThread.submit_callback(self.process_images)

        self.threadPool.start(workerThread)

    def _update_img_container_status(self, data, icon):
        index, result = data
        target_button = self.imageContainers[index].findChild(QPushButton)
        target_button.setIcon(icon)
