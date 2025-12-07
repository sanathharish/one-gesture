# ui/air_pointer.py

from PyQt5 import QtWidgets, QtGui, QtCore


class AirPointer(QtWidgets.QWidget):
    """
    A transparent always-on-top pointer displayed across the entire desktop.
    """

    def __init__(self, size=35, color=(0, 255, 255)):
        super().__init__()

        self.size = size
        self.color = QtGui.QColor(color[0], color[1], color[2], 255)

        # Transparent background, no borders, always on top
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
            | QtCore.Qt.WindowTransparentForInput
        )

        # Enable translucent background
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        # Initial position
        self.x = 500
        self.y = 500

        # Resize widget to fixed circle size
        self.resize(self.size, self.size)

        # Show it
        self.show()

    # --------------------------------------------------------------------
    # UPDATE POSITION (called every frame)
    # --------------------------------------------------------------------
    def update_position(self, x, y):
        """
        Moves the pointer overlay to global screen coords.
        """
        self.x = int(x - self.size / 2)
        self.y = int(y - self.size / 2)
        self.move(self.x, self.y)
        self.update()

    # --------------------------------------------------------------------
    # PAINT EVENT (draw neon circle)
    # --------------------------------------------------------------------
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)

        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Outer glow
        glow_color = QtGui.QColor(self.color)
        glow_color.setAlpha(180)

        pen = QtGui.QPen(glow_color)
        pen.setWidth(6)
        painter.setPen(pen)

        painter.drawEllipse(3, 3, self.size - 6, self.size - 6)

        # Inner circle
        inner_color = QtGui.QColor(self.color)
        inner_color.setAlpha(255)

        pen2 = QtGui.QPen(inner_color)
        pen2.setWidth(3)
        painter.setPen(pen2)

        painter.drawEllipse(6, 6, self.size - 12, self.size - 12)
