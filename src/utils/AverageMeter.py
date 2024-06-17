#!/usr/bin/env python3

# Copyright (c) 2017-present, Philips, Inc.

"""
-------------------------------------------------
   File Name：     AverageMeter
   Description :
   Author :        qizhong.lin@philips.coom
   date：          21-1-10
-------------------------------------------------
   Change Activity:
                   21-1-10:
-------------------------------------------------
"""



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n > 0:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

