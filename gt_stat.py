# businiaoo 2021/1/30 1182693164@qq.com
"""
The area, size, and category distribution of the bounding box in the statistical label
统计标签中边界框的面积、大小、类别分布
"""
import cv2
import os
from numpy import loadtxt, array
from PIL import Image


class LabelInfo:
    def __init__(self, label_path, label_format, img_path=None):
        """
        :param label_path:  str
                            if label_format is ".txt" or ".xml"(one ".txt" or ".xml" file for one image),
                               then label_path is a directory path
                            if label_format is ".json"(like coco format), then label_path is a certain file path
        :param label_format: int
                            1 for [c x y w h] normalized  (yolo format) ".txt" file
                            2 for [c x y w h] not normalized  ".txt" file
                            3 for [c x1 y1 x2 y2] normalized  ".txt" file
                            4 for [c x1 y1 x2 y2] not normalized  ".txt" file
                            5 for [c x1 y1 x2 y2 x3 y3 x4 y4] normalized  ".txt" file
                            6 for [c x1 y1 x2 y2 x3 y3 x4 y4] not normalized  ".txt" file
                            7 for coco format  ".json" file
                            8 for voc format  ".xml" file
        """
        self.all_labels = self.label_conversion(label_path, label_format, img_path)

    @staticmethod
    def label_conversion(label_path, label_format, img_path):
        all_labels = array([])
        if label_format == 1:  # [c x y w h] normalized  (yolo format)
            label_names = os.listdir(label_path)

            for label_name in label_names:

                labels = loadtxt(os.path.join(label_path, label_name))


        return all_labels

