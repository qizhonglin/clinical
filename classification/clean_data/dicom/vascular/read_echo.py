# -*- coding: utf-8 -*-
"""
:Authors: Qizhong Lin <qizhong.lin@philips.com>,
:Copyright: This file contains proprietary information of Philips 
            Innovative Technologies. Copying or reproduction without prior
            written approval is prohibited.

            Philips internal use only - no distribution outside Philips allowed
"""
import os
import scipy.io as sio
import cv2 as cv
import matplotlib.pyplot as plt

from classification.utils.ImageVis import grays2img


def read_echo(echo_file):
    """

    :param echo_file: echo file in .mat format from matlab
    :return: seq with shape (seq num, height, width)
    """
    echo = sio.loadmat(echo_file)

    seq = echo["echo"]
    seq = seq.transpose(2, 0, 1)

    return seq


def mat2pic(echo_file, save_dir, extname='png'):
    if not os.path.isfile(echo_file):
        return

    os.makedirs(save_dir, exist_ok=True)

    seq = read_echo(echo_file)

    for idx, frame in enumerate(seq):
        save_file = os.path.join(save_dir, f'{idx:05}.{extname}')
        cv.imwrite(save_file, frame)


def main():
    MAT_DIR = "/media/qzlin/25793662-6b5a-431d-8402-87c5bd9357df/dataset/Hackthon/vascular/mat"
    echo_file = os.path.join(MAT_DIR, "Philips_p1_VERT_IM_0012.mat")

    seq = read_echo(echo_file)
    # plt.imshow(seq[0], cmap="gray")
    print(f"seq shape: {seq.shape}")

    display_grid = grays2img(seq)
    save_file = os.path.join(MAT_DIR, 'echo.jpg')
    cv.imwrite(save_file, display_grid)





if __name__ == '__main__':
    main()

    plt.show()