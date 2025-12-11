# -*- coding: utf-8 -*-

import cv2 as cv
import os
from tqdm import tqdm
import json

from LymphNode.config import DATA_ROOT as root


def _crop_roi(image_file, mask_file, roi_file, margin=0.5):
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

    box = x_s, y_s, x_e, y_e = int(x_s), int(y_s), int(x_e), int(y_e)

    roi = img[y_s:y_e, x_s:x_e, :]

    #cv.imwrite(roi_file, roi)

    return roi, box


def crop_roi():
    roi_box = {}
    for group in ['huaxi', 'tianfu']:
        roi_box[group] = {}

        dir_images = os.path.join(root, group, "images")
        dir_masks = os.path.join(root, group, "masks")
        dir_rois = os.path.join(root, group, "roi")

        for name in ["malignant", "benign"]:
            roi_box[group][name] = {}

            print('processing {0}/{1}...'.format(group, name))
            dir_images_mode = os.path.join(dir_images, name)
            dir_masks_mode = os.path.join(dir_masks, name)
            dir_rois_mode = os.path.join(dir_rois, name)
            os.makedirs(dir_rois_mode, exist_ok=True)

            for file in tqdm(os.listdir(dir_images_mode)):
                image_file = os.path.join(dir_images_mode, file)
                mask_file = os.path.join(dir_masks_mode, file)
                roi_file = os.path.join(dir_rois_mode, file)

                _, box = _crop_roi(image_file, mask_file, roi_file)

                roi_box[group][name][file] = box

    save_file = os.path.join(root, 'cache', 'roi_box.json')
    json.dump(roi_box, open(save_file, 'w'), indent=4)
    roi_box = json.loads(open(save_file).read())
    print(roi_box)


def crop_echo():
    for group in ['huaxi', 'tianfu']:
        dir_images = os.path.join(root, group, "images")
        dir_rois = os.path.join(root, group, "echo")

        for name in ["malignant", "benign"]:
            print('processing {0}/{1}...'.format(group, name))
            dir_images_mode = os.path.join(dir_images, name)
            dir_rois_mode = os.path.join(dir_rois, name)
            os.makedirs(dir_rois_mode, exist_ok=True)

            for file in tqdm(os.listdir(dir_images_mode)):
                image_file = os.path.join(dir_images_mode, file)
                roi_file = os.path.join(dir_rois_mode, file)

                _crop_echo(image_file, roi_file)


def _crop_echo(image_file, roi_file):
    image = cv.imread(image_file)

    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    # image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

    thresh = image[-1, -1] + 1
    ret, binary = cv.threshold(image, thresh, 255, cv.THRESH_BINARY)

    contours, hier = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        if w * h < 1000: continue
        boxes.append((x, y, w, h, w * h))

    boxes_sorted = sorted(boxes, key=lambda ele: ele[-1], reverse=True)

    box = boxes_sorted[0]
    x_s, y_s, x_e, y_e = int(box[0]), int(box[1]), int(box[0]+box[2]), int(box[1]+box[3])

    image = cv.imread(image_file)
    roi = image[y_s:y_e, x_s:x_e, :]
    cv.imwrite(roi_file, roi)

    return roi





if __name__ == '__main__':
    crop_roi()
    # crop_echo()