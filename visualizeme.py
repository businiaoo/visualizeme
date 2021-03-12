# businiaoo 2021/1/30 businiao006@gmail.com
# https://github.com/businiaoo/visualizeme

from label_statistics.gt_statistics import LabelInfo


if __name__ == "__main__":
    label_path = "../data/labels"
    label_format = 1
    img_path = "../data/images"

    label = LabelInfo(label_path, label_format, img_path)
    a = label.statistic()



