from copy import deepcopy
import os
import sys
from PIL import Image
import numpy as np
import torch
import PyQt5
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QGridLayout, QFileDialog, QCheckBox
from PyQt5.QtGui import QPixmap

from test import load_model, stylized2
from cnn_vis.cnn_visualization import get_filter, gmm
import utils
import args
device = args.device
output_dir = 'data/output'
k = 10


class MainWindow(QWidget):
    def __init__(self, orig_model):
        super(self.__class__, self).__init__()
        self.setupUi()
        self.show()

        self.cwd = os.getcwd()
        self.orig_model = orig_model
        self.img_path = ''
        self.style_id = -1
        self.prep_img = []
        self.orig_image = (0, 0)

        self.style_elements = []
        self.masks = []

    def setupUi(self):
        self.setWindowTitle("Style Transfer")

        # content image
        self.label_c = QLabel()
        self.label_c.setText('Content image')
        self.button_c = QPushButton()
        self.button_c.setText('select')
        self.label_img_c = QLabel()
        # style image
        self.label_s = QLabel()
        self.label_s.setText('Style image')
        self.button_s = QPushButton()
        self.button_s.setText('select')
        self.label_img_s = QLabel()
        # gmm button
        self.button_gmm = QPushButton()
        self.button_gmm.setText('gmm('+str(k)+')')
        # stylized image
        self.label_o = QLabel()
        self.label_o.setText('Stylized image')
        self.button_o = QPushButton()
        self.button_o.setText('stylize...')
        self.label_img_o = QLabel()
        self.label_img_o_path = QLabel()

        ##################
        # set position
        ##################
        grid = QGridLayout()
        grid.setSpacing(10)
        # content image
        grid.addWidget(self.label_c, 1, 0)
        grid.addWidget(self.button_c, 1, 1)
        grid.addWidget(self.label_img_c, 2, 0, 5, 5)
        # style image
        grid.addWidget(self.label_s, 7, 0)
        grid.addWidget(self.button_s, 7, 1)
        grid.addWidget(self.button_gmm, 7, 3)
        grid.addWidget(self.label_img_s, 8, 0, 5, 5)
        # gmm - checkbox
        for idx in range(0, k):
            checkBox = QCheckBox(str(idx+1), self)
            checkBox.id_ = idx
            checkBox.stateChanged.connect(self.checkchoice)
            grid.addWidget(checkBox, 10, 5+2*idx)
        # gmm - label
        for jidx in range(0, k):
            setattr(self, 'label_{}_{}'.format(str(k), jidx+1), QLabel())
            grid.addWidget(getattr(self, 'label_{}_{}'.format(str(k), jidx+1)), 8, 5+2*jidx)
        # stylized image
        grid.addWidget(self.button_o, 13, 0)
        grid.addWidget(self.label_img_o_path, 14, 0, 1, -1)
        grid.addWidget(self.label_img_o, 16, 0, 5, 5)

        self.setLayout(grid)

        ##################
        # button connect
        ##################
        self.button_c.clicked.connect(self.load_c_img)
        self.button_s.clicked.connect(self.load_s_img)
        self.button_o.clicked.connect(self.stylize)
        self.button_gmm.clicked.connect(self.gmm)

    def load_c_img(self):
        c_path = QFileDialog.getOpenFileName(self, 'Select Content Image', self.cwd, '(*.jpg)')
        if c_path != ('', ''):
            c_path = c_path[0]
            self.label_img_c.setPixmap(QPixmap(c_path).scaled(300, 300, PyQt5.QtCore.Qt.KeepAspectRatio))
            self.img_path = c_path
            original_image = Image.open(self.img_path).convert('RGB')
            self.orig_image = (original_image.size[1], original_image.size[0])
            self.prep_img = utils.preprocess_image(original_image)

    def load_s_img(self):
        s_path = QFileDialog.getOpenFileName(self, 'Select Style Image', self.cwd, '(*.jpg)')
        if s_path != ('', ''):
            s_path = s_path[0]
            self.label_img_s.setPixmap(QPixmap(s_path).scaled(300, 300, PyQt5.QtCore.Qt.KeepAspectRatio))
            style_id_string = s_path.split('/')[-1].split('.')[0]
            self.style_id = list(map(int, style_id_string.split(' ')))
            # print(self.style_id)

    def stylize(self):
        img_name = self.img_path.split('/')[-1].split('.')[0] + '_' + str(self.style_id[0]) + '.jpg'
        output_path = output_dir + '/' + img_name
        self.label_img_o_path.setText('Save in \'' + output_path + '\'')
        test_img = self.prep_img.to(device)
        data = torch.zeros([args.batch_size, 3, args.IMG_SIZE, args.IMG_SIZE])
        if self.style_elements == []:   # if not gmm
            output_test = self.orig_model(test_img.expand_as(data), self.style_id)[0].cpu().detach()
            utils.showimg(output_test, False, output_path, self.orig_image)
            self.label_img_o.setPixmap(QPixmap(output_path).scaled(1069, 300, PyQt5.QtCore.Qt.KeepAspectRatio))
        else:
            style_id = str(self.style_id[0])
            total_mean = np.zeros_like(self.masks[0])
            for cid, mask in enumerate(self.masks):
                if cid in self.style_elements:
                    # print(cid)
                    total_mean = total_mean + mask
            total_mean = total_mean.reshape(128, 128, 3, 3)
            # mask = mask[np.newaxis, :]
            total_mean = torch.FloatTensor(total_mean)
            # print(mask.shape)
            save_path = output_dir + '/k/' + 'k' + str(k)
            new_model = deepcopy(orig_model)
            new_model.state_dict()['style_bank.'+style_id+'.0.conv2d.weight'].copy_(total_mean)
            stylized2(self.prep_img, new_model, save_path, [int(style_id)])
            self.label_img_o.setPixmap(QPixmap(save_path+'.png').scaled(300, 300, PyQt5.QtCore.Qt.KeepAspectRatio))

    def gmm(self):
        style_id = str(self.style_id[0])
        f_flatten = get_filter(self.orig_model, 'style_bank.'+ style_id +'.0')

        # clustering
        centers, labels = gmm(f_flatten, k)

        self.masks = np.zeros(shape=(k, f_flatten.shape[0], f_flatten.shape[1]))
        for cid in range(k):
            for pid in range(f_flatten.shape[0]):
                if labels[pid] == cid:
                    self.masks[cid][pid] = centers[cid]
        # plot
        for cid, mask in enumerate(self.masks):
            mask = mask.reshape(128, 128, 3, 3)
            # mask = mask[np.newaxis, :]
            mask = torch.FloatTensor(mask)
            # print(mask.shape)
            save_path = output_dir + '/k/' + 'k' + str(k) + '_c' + str(cid)

            # stylized
            new_model = deepcopy(orig_model)
            new_model.state_dict()['style_bank.'+style_id+'.0.conv2d.weight'].copy_(mask)
            # style_id = list(map(int, style_id.split(' ')))
            stylized2(self.prep_img, new_model, save_path, [int(style_id)])
            del new_model
        for jidx in range(0, k):
            getattr(self, 'label_{}_{}'.format(str(k), jidx+1)).setPixmap(QPixmap(save_path.split('_c')[0]+'_c'+str(jidx)+'.png').scaled(80, 80, PyQt5.QtCore.Qt.KeepAspectRatio))

    def checkchoice(self, state):
        checkBox = self.sender()
        if state == PyQt5.QtCore.Qt.Unchecked:
            self.style_elements.remove(checkBox.id_)
        elif state == PyQt5.QtCore.Qt.Checked:
            self.style_elements.append(checkBox.id_)
        print(self.style_elements)


if __name__ == '__main__':
    orig_model = load_model()

    app = QApplication(sys.argv)
    MainWindow = MainWindow(orig_model)
    sys.exit(app.exec_())
