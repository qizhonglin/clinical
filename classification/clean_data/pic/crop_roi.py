# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import os

print(cv.__version__)


def crop_roi(image_file, mask_file, roi_file, margin=0.5):
    img = cv.imread(image_file)
    mask = cv.imread(mask_file)
    height, width, _ = mask.shape

    edge = cv.Canny(mask, 25, 255)
    contours, hierarchy = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    area_contour = [(cv.contourArea(c), c) for c in contours]
    area_contour.sort(key=lambda ele: ele[0], reverse=True)
    (x, y, w, h) = cv.boundingRect(area_contour[0][1])

    x_s = max(x - margin * w, 0)
    y_s = max(y - margin * h, 0)
    x_e = min(x + w + margin * w, width-1)
    y_e = min(y + h + margin * h, height-1)

    x_s, y_s, x_e, y_e = int(x_s), int(y_s), int(x_e), int(y_e)

    roi = img[y_s:y_e, x_s:x_e, :]

    cv.imwrite(roi_file, roi)

    return roi


def main():
    from src.config import DATA_ROOT as root

    for group in ['huaxi', 'tianfu']:
        dir_images = os.path.join(root, group, "images")
        dir_masks = os.path.join(root, group, "masks")
        dir_rois = os.path.join(root, group, "roi")

        for name in ["malignant", "benign"]:
            print('processing {0}/{1}...'.format(group, name))
            dir_images_mode = os.path.join(dir_images, name)
            dir_masks_mode = os.path.join(dir_masks, name)
            dir_rois_mode = os.path.join(dir_rois, name)
            os.makedirs(dir_rois_mode, exist_ok=True)

            for file in os.listdir(dir_images_mode):
                image_file = os.path.join(dir_images_mode, file)
                mask_file = os.path.join(dir_masks_mode, file)
                roi_file = os.path.join(dir_rois_mode, file)

                crop_roi(image_file, mask_file, roi_file)




if __name__ == '__main__':
    main()