from PyQt5 import QtWidgets, QtCore
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from PyQt5.QtCore import QTimer, QTime, endl
import sys  # We need sys so that we can pass argv to QApplication
import os
import numpy as np







class MainWindow(QtWidgets.QMainWindow):










    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        #self.graphWidget = pg.PlotWidget()
        #pg.setConfigOptions(antialias=True)

        #self.graphWidget.setTitle("<span style=\"color:blue;font-size:30pt\">seconds per frame</span>")
        #self.graphWidget.setLabel('left', "<span style=\"color:red;font-size:20px\">Time (ms)</span>")
        #self.graphWidget.setLabel('bottom', "<span style=\"color:red;font-size:20px\">Frame (n)</span>")
        #self.graphWidget.showGrid(x=True, y=True)

        win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
        win.resize(1000,600)
        win.setWindowTitle('pyqtgraph example: Plotting')

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        self.p1 = win.addPlot()
        self.p1.setTitle("<span style=\"color:blue;font-size:30pt\">seconds per frame</span>")
        self.p1.setLabel('left', "<span style=\"color:red;font-size:20px\">Time (ms)</span>")
        self.p1.setLabel('bottom', "<span style=\"color:red;font-size:20px\">Frame (n)</span>")
        self.p1.showGrid(x=True, y=True)

        self.p2 = win.addPlot()
        self.p2.setTitle("<span style=\"color:blue;font-size:30pt\">frame per second</span>")
        self.p2.setLabel('left', "<span style=\"color:red;font-size:20px\">FPS (n/s)</span>")
        self.p2.setLabel('bottom', "<span style=\"color:red;font-size:20px\">Frame (n/s)</span>")
        self.p2.showGrid(x=True, y=True)

        self.setCentralWidget(win)



        # plot data: x, y values
        #self.graphWidget.plot(hour, temperature)
        self.n = np.array([0])
        self.t = np.array([0])
        self.fps = np.array([0])

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(250)

    def tick(self):
        self.p1.clear()
        self.p2.clear()


        #print(self.n)
        pen = pg.mkPen(color=(255, 0, 0), width=5, style=QtCore.Qt.DashLine)
        self.p1.plot(self.n,self.t,symbol='+', symbolSize=15, symbolBrush=('b'))
        self.p2.plot(self.n,self.fps,symbol='o', symbolSize=15, symbolBrush=('b'))

    @QtCore.pyqtSlot(float,int)
    def readData(self,time,count):
        self.n = np.append(self.n,count)
        self.t = np.append(self.t,time*1000)
        self.fps = np.append(self.fps,1/time)
        #print(self.t)
        if len(self.n) > 480:
            self.n = self.n[1:]
            self.t = self.t[1:]
            self.fps = self.fps[1:]

    @QtCore.pyqtSlot(bool)
    def pauseResume(self,PR):
        if PR:
            self.timer.stop()
        else:
            self.timer.start(250)





def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()