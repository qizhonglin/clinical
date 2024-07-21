# -*- coding: utf-8 -*-
"""
:Authors: Qizhong Lin <qizhong.lin@philips.com>,
:Copyright: This file contains proprietary information of Philips 
            Innovative Technologies. Copying or reproduction without prior
            written approval is prohibited.

            Philips internal use only - no distribution outside Philips allowed
"""
import random
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import functional as F
from torch import Tensor


class RandomCenterCrop(object):
    def __init__(self, margin_ratio=0.5 / 4):
        self.margin_ratio = margin_ratio

    def __call__(self, tensor):
        _, h, w = tensor.shape
        x1 = random.randint(0, int(self.margin_ratio * w))
        y1 = random.randint(0, int(self.margin_ratio * h))
        cropped = tensor[:, y1:h - y1, x1:w - x1]
        return cropped



class RandomTopCrop(object):
    def __init__(self, area_scale=[0.8, 1.0]):
        self.area_scale = area_scale

    @staticmethod
    def get_params(img: Tensor, area_scale: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random top crop.

        Args:
            img (Tensor): Input image.
            area_scale (list): range of ratio of the origin size cropped

        Returns:
            tuple: params (top, left, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        _, height, width = F.get_dimensions(img)
        area = height * width
        aspect_ratio = width / height

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(area_scale[0], area_scale[1]).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                top = 0
                left = torch.randint(0, width - w + 1, size=(1,)).item()
                return top, left, h, w

        return 0, 0, height, width

    def __call__(self, img: Tensor):
        y1, x1, h, w = self.get_params(img, self.area_scale)
        cropped = img[:, y1:h - y1, x1:w - x1]
        return cropped

