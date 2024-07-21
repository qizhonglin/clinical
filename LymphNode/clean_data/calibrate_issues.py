# -*- coding: utf-8 -*-

import os

from LymphNode.config import DATA_ROOT as root



issues = [
    "5485110",
    "6169843",
    "6260363",
    "6275209",
    "6300941",
    "6346214",
    "6406025",
    "6777583",
    "2012090018",
    "5187898",
    "6150225",
    "2012110012",

    "2206150153",            # tianfu
    "2304210237"
]

def main():
    for group in ['huaxi', 'tianfu']:
        dir_images = os.path.join(root, group, "images")
        dir_masks = os.path.join(root, group, "masks")
        dir_rois = os.path.join(root, group, "roi")

        for name in ["malignant", "benign"]:
            print('processing {0}/{1}...'.format(group, name))
            dir_images_mode = os.path.join(dir_images, name)
            dir_masks_mode = os.path.join(dir_masks, name)
            dir_rois_mode = os.path.join(dir_rois, name)

            for file in os.listdir(dir_images_mode):
                filename, file_extension = os.path.splitext(file)
                if filename in issues:
                    roi_file = os.path.join(dir_rois_mode, file)
                    if os.path.exists(roi_file):
                        os.remove(roi_file)
                        print(f"The file {roi_file} has been deleted.")


if __name__ == '__main__':
    main()