# businiaoo 2021/1/30 1182693164@qq.com
import gt_stat


if __name__ == "__main__":
    label_path = "../data/labels"
    label_format = 1
    img_path = "../data/images"

    label = gt_stat.LabelInfo(label_path, label_format, img_path)


