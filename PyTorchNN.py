import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import sys  # sys нужен для передачи argv в QApplication
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from appdesign import Ui_MainWindow
import matplotlib.pyplot as plt
from settingswinClass import SettingsWin
from aboutwinClass import AboutWin
import imagesize
import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure



class CNN1(torch.nn.Module):  # 1C1P2F

    def __init__(self):
        super(CNN1, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)

        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = x.view(-1, 18 * 16 * 16)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x


class CNN2(torch.nn.Module):  # 2C1P2F

    def __init__(self):
        super(CNN2, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(18, 108, kernel_size=3, stride=1, padding=1)

        self.fc1 = torch.nn.Linear(108 * 16 * 16, 64)

        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = x.view(-1, 108 * 16 * 16)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x


class CNN3(torch.nn.Module):  # 1C2P2F

    def __init__(self):
        super(CNN3, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(18 * 8 * 8, 64)

        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = self.pool(x)

        x = self.pool2(x)

        x = x.view(-1, 18 * 8 * 8)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x


class CNN4(torch.nn.Module):  # 2C2P2F

    def __init__(self):
        super(CNN4, self).__init__()

        self.activated_features = None

        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(18, 108, kernel_size=3, stride=1, padding=1)

        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(108 * 8 * 8, 64)

        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)


        x = F.relu(x)

        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.pool2(x)

        x = x.view(-1, 108 * 8 * 8)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x

def createLossAndOptimizer(net, learning_rate=0.0001):  # создание функции потерь и оптимизатора

    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    return (loss, optimizer)


def graphics(loss_l, acc_l, loss_l_v, acc_l_v):  # графическое описание архитектуры (точность, потеря)

    fig = plt.figure("Информация")
    axes = fig.add_subplot(122)
    axes.set_title("Потеря")
    axes.plot(np.array(loss_l))
    axes.plot(np.array(loss_l_v))
    axes.set_xlabel("эпоха")
    plt.legend(['обучение', 'проверка'], loc='upper right')
    axes = fig.add_subplot(121)
    axes.set_title("Точность")
    axes.plot(np.array(acc_l))
    axes.plot(np.array(acc_l_v))
    axes.set_xlabel("эпоха")
    plt.legend(['обучение', 'проверка'], loc='upper right')
    fig.show()
    fig.savefig('1C2P2F')


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, data):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        res = ax.imshow(data, cmap='jet',
                        interpolation='nearest')
        self.draw()


def translateClass(predicted):  # Перевод на русский язык результата анализа

    str_class = None

    if classes[int(predicted[0])] == 'truck':
        str_class = 'грузовик'

    if classes[int(predicted[0])] == "plane":
        str_class = 'самолет'

    if classes[int(predicted[0])] == 'frog':
        str_class = 'лягушка'

    if classes[int(predicted[0])] == 'cat':
        str_class = 'кот'

    if classes[int(predicted[0])] == 'dog':
        str_class = 'собака'

    if classes[int(predicted[0])] == 'car':
        str_class = 'машина'

    if classes[int(predicted[0])] == 'bird':
        str_class = 'птица'

    if classes[int(predicted[0])] == 'deer':
        str_class = 'олень'

    if classes[int(predicted[0])] == 'horse':
        str_class = 'лошадь'

    if classes[int(predicted[0])] == 'ship':
        str_class = 'корабль'

    return str_class


class Application(QtWidgets.QMainWindow):  # клас оконного приложения

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.action_2.triggered.connect(self.opensettingsWin)
        self.ui.action.triggered.connect(self.openaboutWin)
        self.ui.pushButton.clicked.connect(self.pushButtonClicked)
        self.settingsWin = None
        self.aboutWin = None
        self.ui.openFile.triggered.connect(self.open_image)
        self.filename = None
        self.img_pixmap = None
        self.label_image = None
        self.loaded = False
        self.CNN = CNN4()
        self.CNN.load_state_dict(torch.load('/Users/mikhaildelba/PycharmProjects/ScuteN/model4'))
        self.r1 = False
        self.r2 = False
        self.r3 = False
        self.r4 = True
        self.pc = PlotCanvas(self, width=3, height=2)
        self.pc.move(725,12)
        self.pc.resize(0,0)

    def open_image(self): # загрузка изображения
        self.loaded = False
        try:
            self.filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', "Image files (*.jpg *.gif *.png)")[0]

            if self.label_image is not None:
                self.label_image.close()
            self.label_image = QtWidgets.QLabel(self.ui.graphicsView)
            self.label_image.clear()
            self.label_image = QtWidgets.QLabel(self.ui.graphicsView)
            self.image_fit()
            self.label_image.show()
            self.ui.label.clear()
            self.loaded = True
        finally:
            return 0

    def pushButtonClicked(self): # кнопка запуска
        if self.loaded != False:
            cv2im = cv2.imread(self.filename)
            cv2im = pre_image(cv2im)
            outputs = self.CNN(cv2im)
            _, predicted = torch.max(outputs, 1)
            str_class = translateClass(predicted)
            self.ui.label.setText('На картинке ' + str_class)
            outs = outputs.data.cpu().numpy()  ### convert to numpy
            self.pc.resize(300,300)
            self.pc.plot(outs)

    def image_fit(self): # отображение изображения
        if self.filename is not None:
            w, h = imagesize.get(self.filename)
            if w > h:
                self.label_image.setPixmap(QtGui.QPixmap.fromImage(QtGui.QImage(self.filename).scaledToWidth(self.ui.graphicsView.width())))
            else:
                self.label_image.setPixmap(QtGui.QPixmap.fromImage(QtGui.QImage(self.filename).scaledToHeight(self.ui.graphicsView.height())))

    def resizeEvent(self, QResizeEvent):
        self.ui.graphicsView.heightForWidth(self.ui.graphicsView.width())

    def closeEvent(self, event):  # событие: закрытие всех окон

        if self.settingsWin is not None:
            self.settingsWin.close()

        if self.aboutWin is not None:
            self.aboutWin.close()

    def showInfo (self):

        self.settingsWin.label_image = QtWidgets.QLabel(self.settingsWin.ui.graphicsView)
        if self.settingsWin.ui.radioButton.isChecked():
            self.settingsWin.filename = '/Users/mikhaildelba/PycharmProjects/ScuteN/1C1P2F.png'
        if self.settingsWin.ui.radioButton_2.isChecked():
            self.settingsWin.filename = '/Users/mikhaildelba/PycharmProjects/ScuteN/2C1P2F.png'
        if self.settingsWin.ui.radioButton_3.isChecked():
            self.settingsWin.filename = '/Users/mikhaildelba/PycharmProjects/ScuteN/1C2P2F.png'
        if self.settingsWin.ui.radioButton_4.isChecked():
            self.settingsWin.filename = '/Users/mikhaildelba/PycharmProjects/ScuteN/2C2P2F.png'
        self.settingsWin.label_image.setPixmap(
            QtGui.QPixmap.fromImage(QtGui.QImage(self.settingsWin.filename).scaledToWidth(self.settingsWin.ui.graphicsView.width())))
        self.settingsWin.label_image.show()

    def opensettingsWin(self):
        self.settingsWin = SettingsWin()
        self.settingsWin.ui.pushButton.clicked.connect(self.setArc)
        self.settingsWin.ui.radioButton.setChecked(self.r1)
        self.settingsWin.ui.radioButton_2.setChecked(self.r2)
        self.settingsWin.ui.radioButton_3.setChecked(self.r3)
        self.settingsWin.ui.radioButton_4.setChecked(self.r4)
        self.settingsWin.ui.radioButton.toggled.connect(self.showInfo)
        self.settingsWin.ui.radioButton_2.toggled.connect(self.showInfo)
        self.settingsWin.ui.radioButton_3.toggled.connect(self.showInfo)
        self.settingsWin.ui.radioButton_4.toggled.connect(self.showInfo)
        self.showInfo()
        self.settingsWin.show()

    def openaboutWin(self):
        self.aboutWin = AboutWin()
        self.aboutWin.show()

    def setArc(self): # создание сети с нужной архитектурой

        if self.settingsWin.ui.radioButton.isChecked():
            self.r1 = True
            self.r2 = False
            self.r3 = False
            self.r4 = False
            self.CNN = CNN1()
            self.CNN.load_state_dict(torch.load('/Users/mikhaildelba/PycharmProjects/ScuteN/model'))

        if self.settingsWin.ui.radioButton_2.isChecked():
            self.r2 = True
            self.r1 = False
            self.r3 = False
            self.r4 = False
            self.CNN = CNN2()
            self.CNN.load_state_dict(torch.load('/Users/mikhaildelba/PycharmProjects/ScuteN/model2'))

        if self.settingsWin.ui.radioButton_3.isChecked():

            self.r3 = True
            self.r1 = False
            self.r2 = False
            self.r4 = False
            self.CNN = CNN3()
            self.CNN.load_state_dict(torch.load('/Users/mikhaildelba/PycharmProjects/ScuteN/model3'))

        if self.settingsWin.ui.radioButton_4.isChecked():

            self.r4 = True
            self.r1 = False
            self.r2 = False
            self.r3 = False
            self.CNN = CNN4()
            self.CNN.load_state_dict(torch.load('/Users/mikhaildelba/PycharmProjects/ScuteN/model4'))

        self.settingsWin.close()
        return self.CNN



seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


def pre_image(cv2im, resize_im=True):  # предварительная работа над изображением

    if resize_im:
        cv2im = cv2.resize(cv2im, (32, 32))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)
    im_as_ten = torch.from_numpy(im_as_arr).float()
    im_as_ten.unsqueeze_(0)
    im_as_var = Variable(im_as_ten, requires_grad=True)  # приведение в переменную
    return im_as_var


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

app = QtWidgets.QApplication(sys.argv)
window = Application()
window.show()
app.exec_()
