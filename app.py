import sys
from pathlib import Path

import imageio.v3 as iio
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtGui

from seam_carving import remove_columns


def np_array2Qimage(img):
    height, width, channel = img.shape
    bytesPerLine = 3 * width
    qImg = QtGui.QImage(
        img, width, height, bytesPerLine, QtGui.QImage.Format.Format_RGB888
    )
    return qImg


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Images
        images = self.createImages()
        # Options
        options = self.createOptions()

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(images)
        main_layout.addLayout(options)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.image_path = "Castle5565.jpg"
        self.image_changed()

    def valueChanged(self, sb):
        im = iio.imread(self.image_path)
        im = remove_columns(im, sb.value())
        print(im.shape)
        qimg = np_array2Qimage(im)

        self.img2.setPixmap(QtGui.QPixmap(qimg))

    def createOptions(self):
        options = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel("number of collumns to be excluded")
        options.addWidget(label)
        spin = pg.SpinBox(value=0, int=True, bounds=[0, 1000], finite=True)
        self.spin = spin
        options.addWidget(spin)
        spin.sigValueChanged.connect(self.valueChanged)
        chooseImageButton = QtWidgets.QPushButton("Choose file")
        chooseImageButton.clicked.connect(self.showDialog)
        options.addWidget(chooseImageButton)
        return options

    def createImages(self):
        images = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        img1 = QtWidgets.QLabel()
        self.img1 = img1
        layout.addWidget(img1)

        img2 = QtWidgets.QLabel()
        self.img2 = img2
        layout.addWidget(img2)
        images.setLayout(layout)

        return images

    def showDialog(self):
        file_dialog = QtWidgets.QFileDialog(self)
        # TODO filter to only images
        # file_dialog.setNameFilter("Text files (*.txt);;All files (*)")

        if file_dialog.exec():
            selected_file = file_dialog.selectedFiles()[0]
            self.image_path = Path(selected_file)
            self.image_changed()

    def image_changed(self):
        im = iio.imread(self.image_path)
        qimg = np_array2Qimage(im)

        self.img1.setPixmap(QtGui.QPixmap(qimg))

        im = remove_columns(im, self.spin.value())
        qimg = np_array2Qimage(im)
        self.img2.setPixmap(QtGui.QPixmap(qimg))


def main() -> None:
    app = pg.mkQApp("Plotting Example")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
