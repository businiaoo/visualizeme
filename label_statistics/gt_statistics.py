# businiaoo 2021/1/30 businiao006@gmail.com
# https://github.com/businiaoo/visualizeme

"""
The area, size, and category distribution of the bounding box in the statistical label
统计标签中边界框的面积、大小、类别分布
"""
import os
# from numpy import loadtxt, array, concatenate, expand_dims, zeros
import numpy as np
from PIL.Image import open
import time

img_formats = ["jpg", "bmp", "jpeg", "tif", "tiff", "png"]


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
        assert len(self.all_labels) != 0, "no label was found"

    def statistic(self):
        categories_distribution = self.category_statistics()
        self.bbox_statistics()
        return categories_distribution, 1

    def category_statistics(self):
        categories_distribution = {}
        categories = self.all_labels[:, 0]
        unique_categories = np.unique(categories)
        for i in range(len(unique_categories)):
            categories_distribution[int(unique_categories[i])] = np.sum(categories == unique_categories[i])
        return categories_distribution

    def bbox_statistics(self):
        def line_length(a, b):
            length = np.sqrt((a[:, 0] - b[:, 0]) ** 2 + (a[:, 1] - b[:, 1]) ** 2)
            return length
        bbox_area_distribution = {}
        bbox_aspect_ratio_distribution = {}
        categories_bbox_area_distribution = {}  # Various categories
        categories_bbox_aspect_ratio_distribution = {}  # Various categories
        categories = self.all_labels[:, 0]
        unique_categories = np.unique(categories)
        for i in range(len(unique_categories)):
            single_category_labels = self.all_labels[categories == unique_categories[i]]
            # Use Helen formula to calculate area
            point1 = single_category_labels[:, 1:3]
            point2 = single_category_labels[:, 3:5]
            point3 = single_category_labels[:, 5:7]
            point4 = single_category_labels[:, 7:]
            length1 = line_length(point1, point2)
            length2 = line_length(point2, point3)
            length3 = line_length(point3, point4)
            length4 = line_length(point4, point1)
            length_diagonal = line_length(point1, point3)
            p1 = (length1 + length2 + length_diagonal) / 2
            p2 = (length3 + length4 + length_diagonal) / 2
            area1 = np.sqrt(p1 * (p1 - length1) * (p1 - length2) * (p1 - length_diagonal))
            area2 = np.sqrt(p2 * (p2 - length3) * (p2 - length4) * (p2 - length_diagonal))
            area = area1 + area2
            print(area)

    @staticmethod
    def label_conversion(label_path, label_format, img_path):
        all_labels = np.array([])
        t0 = time.time()

        if label_format in [1, 3, 5]:  # normalized
            file_names = os.listdir(label_path)
            label_names = [single_name for single_name in file_names if single_name.endswith(".txt")]

            file_names = os.listdir(img_path)
            img_names = [single_name for single_name in file_names if single_name.split(".")[-1].lower() in img_formats]

            for label_name in label_names:

                labels = np.loadtxt(os.path.join(label_path, label_name))
                if len(labels) == 0:
                    continue
                else:
                    if len(labels.shape) == 1:
                        labels = np.expand_dims(labels, axis=0)
                    img_name = [a for a in img_names if a.startswith(label_name.split(".")[0])]
                    if len(img_name):
                        img_name = img_name[0]
                    else:
                        print("image not found.")
                        continue
                    img = open(os.path.join(img_path, img_name))
                    w, h = img.size
                    new_labels = np.zeros((labels.shape[0], 9))
                    new_labels[:, 0] = labels[:, 0]
                    new_labels[:, 1] = (labels[:, 1] - labels[:, 3] / 2) * w
                    new_labels[:, 2] = (labels[:, 2] - labels[:, 4] / 2) * h
                    new_labels[:, 3] = (labels[:, 1] + labels[:, 3] / 2) * w
                    new_labels[:, 4] = (labels[:, 2] - labels[:, 4] / 2) * h
                    new_labels[:, 5] = (labels[:, 1] + labels[:, 3] / 2) * w
                    new_labels[:, 6] = (labels[:, 2] + labels[:, 4] / 2) * h
                    new_labels[:, 7] = (labels[:, 1] - labels[:, 3] / 2) * w
                    new_labels[:, 8] = (labels[:, 2] + labels[:, 4] / 2) * h
                    if len(all_labels) == 0:
                        all_labels = new_labels
                    else:
                        all_labels = np.concatenate([all_labels, new_labels], axis=0)
        print(time.time()-t0)
        return all_labels
