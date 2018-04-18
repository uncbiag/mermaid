#!/usr/bin/env python3

from __future__ import print_function
from builtins import str
from builtins import object
__author__ = "Ashwin Nanjappa"

# GUI viewer to view JSON data as tree.
# Ubuntu packages needed:
# python3-pyqt5

# Std
import argparse
import collections
import json
import sys

# External
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

class TextToTreeItem(object):

    def __init__(self):
        self.text_list = []
        self.titem_list = []

    def append(self, text_list, titem):
        for text in text_list:
            self.text_list.append(text)
            self.titem_list.append(titem)

    # Return model indices that match string
    def find(self, find_str):

        titem_list = []
        for i, s in enumerate(self.text_list):
            if find_str in s:
                titem_list.append(self.titem_list[i])

        return titem_list


class JsonView(QtWidgets.QWidget):

    def __init__(self, fpath):
        super(JsonView, self).__init__()

        self.find_box = None
        self.tree_widget = None
        self.text_to_titem = TextToTreeItem()
        self.find_str = ""
        self.found_titem_list = []
        self.found_idx = 0

        jfile = open(fpath)
        jdata = json.load(jfile, object_pairs_hook=collections.OrderedDict)

        # Find UI

        find_layout = self.make_find_ui()

        # Tree

        self.tree_widget = QtWidgets.QTreeWidget()
        self.tree_widget.setHeaderLabels(["Key", "Value"])
        self.tree_widget.header().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        root_item = QtWidgets.QTreeWidgetItem(["Root"])
        self.recurse_jdata(jdata, root_item)
        self.tree_widget.addTopLevelItem(root_item)

        #root_item.setFlags(root_item.flags() | QtCore.Qt.ItemIsEditable)
        
        # Add table to layout

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.tree_widget)

        # Group box

        gbox = QtWidgets.QGroupBox(fpath)
        gbox.setLayout(layout)

        layout2 = QtWidgets.QVBoxLayout()
        layout2.addLayout(find_layout)
        layout2.addWidget(gbox)

        self.setLayout(layout2)

    def make_find_ui(self):

        # Text box
        self.find_box = QtWidgets.QLineEdit()
        self.find_box.returnPressed.connect(self.find_button_clicked)

        # Find Button
        find_button = QtWidgets.QPushButton("Find")
        find_button.clicked.connect(self.find_button_clicked)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.find_box)
        layout.addWidget(find_button)

        return layout

    def find_button_clicked(self):

        find_str = self.find_box.text()

        # Very common for use to click Find on empty string
        if find_str == "":
            return

        # New search string
        if find_str != self.find_str:
            self.find_str = find_str
            self.found_titem_list = self.text_to_titem.find(self.find_str)
            self.found_idx = 0
        else:
            item_num = len(self.found_titem_list)
            self.found_idx = (self.found_idx + 1) % item_num

        self.tree_widget.setCurrentItem(self.found_titem_list[self.found_idx])


    def recurse_jdata(self, jdata, tree_widget):

        if isinstance(jdata, dict):
            for key, val in list(jdata.items()):
                self.tree_add_row(key, val, tree_widget)
        elif isinstance(jdata, list):
            for i, val in enumerate(jdata):
                key = str(i)
                self.tree_add_row(key, val, tree_widget)
        else:
            print("This should never be reached!")

    def tree_add_row(self, key, val, tree_widget):

        text_list = []

        if isinstance(val, dict) or isinstance(val, list):
            text_list.append(key)
            row_item = QtWidgets.QTreeWidgetItem([key])
            self.recurse_jdata(val, row_item)
        else:
            text_list.append(key)
            text_list.append(str(val))
            row_item = QtWidgets.QTreeWidgetItem([key, str(val)])
            #row_item.setFlags(row_item.flags() | QtCore.Qt.ItemIsEditable)

        tree_widget.addChild(row_item)
        self.text_to_titem.append(text_list, row_item)


class JsonViewer(QtWidgets.QMainWindow):

    def __init__(self):
        super(JsonViewer, self).__init__()

        fpath = sys.argv[1]
        json_view = JsonView(fpath)

        self.setCentralWidget(json_view)
        self.setWindowTitle("JSON Viewer")
        self.show()

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()


def main():
    qt_app = QtWidgets.QApplication(sys.argv)
    json_viewer = JsonViewer()
    sys.exit(qt_app.exec_())


if "__main__" == __name__:
    main()
