#!/usr/bin/env python3

# Copyright (c) 2017-present, Philips, Inc.

"""
-------------------------------------------------
   File Name：     ImageVis
   Description :
   Author :        qizhong.lin@philips.coom
   date：          20-4-9
-------------------------------------------------
   Change Activity:
                   20-4-9:
-------------------------------------------------
"""
import os

import cv2 as cv
import numpy as np


def draw_boxes(img, boxes, color=(0, 255, 0), thickness=1):
    '''
    draw boxes on image

    :param img: gray or color image
    :param boxes: [[x1, y1, x2, y2, ...]]
    :param color:
    :param thickness:
    :return: color image
    '''
    img = img.astype('uint8')
    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    for box in boxes:
        box = [int(b) for b in box]
        left_top = tuple(box[:2])
        right_bottom = tuple(box[2:])
        cv.rectangle(img, left_top, right_bottom, color=color, thickness=thickness)

    return img


def show_seq(seq):
    for idx, frame in enumerate(seq):
        cv.imshow('', frame.astype('uint8'))
        cv.waitKey(100)


def seq2img(seq, fps, box_lesion_seq=None, box_lesion_small_seq=None, box_tissue_seq=None, offset=16):
    '''
    convert seq into large image, via sampling frame in one second

    :param seq: numpy with size (frame_num, height, width, 3)
    :param fps: frames per second, for sampling
    :param box_lesion_seq:
    :param box_lesion_small_seq:
    :param box_tissue_seq:
    :param offset: to skip the first frames within offset second
    :return:
    '''
    images_per_row = 16
    num_fps = len(seq) // fps
    num_fps -= offset
    n_rows = int(np.ceil(num_fps / images_per_row))
    print((n_rows, images_per_row))
    height = 128
    width = int(seq.shape[2] * 128. / seq.shape[1])
    display_grid = np.zeros((height * n_rows, images_per_row * width, 3), dtype='uint8')
    for row in range(n_rows):
        for col in range(images_per_row):
            index = row * images_per_row + col + offset
            if index * fps >= len(seq): continue

            frame_idx = int(index*fps)

            image = seq[frame_idx]

            if box_lesion_seq is not None:
                box_lesion = box_lesion_seq[frame_idx]
                image = draw_boxes(image, [box_lesion], color=(0, 255, 0), thickness=2)

            if box_lesion_small_seq is not None:
                box_lesion_small = box_lesion_small_seq[frame_idx]
                image = draw_boxes(image, [box_lesion_small], color=(0, 0, 255), thickness=2)

            if box_tissue_seq is not None:
                box_tissue = box_tissue_seq[frame_idx]
                image = draw_boxes(image, [box_tissue], color=(255, 0, 0), thickness=2)

            # cv.imshow(str(idx*fps), image)
            new_image = cv.resize(image, (width, height))
            new_image = new_image.astype('uint8')
            # cv.imshow('', new_image)
            # cv.waitKey()
            display_grid[row * height: (row + 1) * height, col * width:(col + 1) * width] = new_image

    return display_grid


def grays2img(seq, fps=1, images_per_row=12):
    num_fps = len(seq) // fps
    n_rows = int(np.ceil(num_fps / images_per_row))
    print((n_rows, images_per_row))

    height = 128
    width = int(seq.shape[2] * 128. / seq.shape[1])
    display_grid = np.zeros((height * n_rows, width*images_per_row), dtype='uint8')
    for row in range(n_rows):
        for col in range(images_per_row):
            index = row * images_per_row + col
            if index * fps >= len(seq): continue

            frame_idx = int(index * fps)

            image = seq[frame_idx]

            new_image = cv.resize(image, (width, height))
            new_image = new_image.astype('uint8')

            display_grid[row * height: (row + 1) * height, col * width:(col + 1) * width] = new_image

    return display_grid


if __name__ == '__main__':
    def get_seq(seq_dir):
        seq = [os.path.join(seq_dir, img) for img in os.listdir(seq_dir)]
        seq.sort()
        return seq

    seq = get_seq('/media/qzlin/25793662-6b5a-431d-8402-87c5bd9357df/dataset/CEUS/outplane/imagesTs/579117_2082')[:600]
    seq += get_seq('/media/qzlin/25793662-6b5a-431d-8402-87c5bd9357df/dataset/CEUS/outplane/imagesTs/568042_1077')[:600]
    print(seq)
    seq = [cv.imread(img) for img in seq]
    H, W = seq[0].shape[0], seq[0].shape[1]
    seq = [cv.resize(img, (W, H)) for img in seq]
    seq = np.asarray(seq)
    display_grid = seq2img(seq, fps=15, offset=0)
    save_file = os.path.join('/home/qzlin/Downloads', '579117_568042.jpg')
    cv.imwrite(save_file, display_grid)
    cv.imshow('', display_grid)
    cv.waitKey()