#!/usr/bin/env python3

# Copyright (c) 2017-present, Philips, Inc.

"""
-------------------------------------------------
   File Name：     DicomReader
   Description :
   Author :        qizhong.lin@philips.coom
   date：          20-3-6
-------------------------------------------------
   Change Activity:
                   20-3-6:
-------------------------------------------------
"""
import os
import pickle
import pydicom
import numpy as np
import cv2 as cv
# from cv2 import VideoWriter_fourcc
from pathlib import Path
from PIL import Image
# import mmcv
from datetime import datetime

import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import convert_color_space


class DicomReader(object):
    def __init__(self, dcm_file):
        self.dcm_file = dcm_file
        self.ds = None

    def read_header(self, element_size='1 KB'):
        '''
        ds.Rows, ds.Columns
        ds.PixelSpacing
        ds.PhotometricInterpretation
        ds.ManufacturerModelName
        ds.PatientName
        ds.PatientID
        ds.UltrasoundColorDataPresent       0: b-mode, 1: doppler

        self.NumberOfFrames
        '''

        with pydicom.dcmread(self.dcm_file, defer_size=element_size) as ds:
            self.ds = ds

        self.PatientName = str(self.ds.PatientName)
        self.PatientID = str(self.ds.PatientID)
        self.Manufacturer = ds.Manufacturer
        self.PhotometricInterpretation = ds.PhotometricInterpretation
        # self.InstitutionName = ds.InstitutionName

        return self

    def read_data(self):
        '''
        :return: dicom data in ndarray
        '''
        self.read_header()
        try:
            # print('load dicom...')
            data = self.ds.pixel_array
            # print('dicom has been loaded!')

            seq = []
            for frame in data:
                if self.PhotometricInterpretation == 'YBR_FULL_422':
                    frame = cv.cvtColor(frame, cv.COLOR_YUV2RGB)
                    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                if self.PhotometricInterpretation == 'RGB':
                    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                # cv.imwrite('frame_cv.jpg', frame)
                # Image.fromarray(frame).save('frame_pil.jpg')
                # plt.imshow(frame)
                # plt.show()
                seq.append(frame)
            data = np.array(seq)
        except:
            data = None
        return data

    def get_framenum(self):
        framenum = 1
        if hasattr(self.ds, 'NumberOfFrames'):
            framenum = self.ds["NumberOfFrames"].value
        return framenum

    def get_fps(self):
        frame_time = None

        if hasattr(self.ds, 'FrameTime'):
            frame_time = self.ds.FrameTime
        if hasattr(self.ds, 'FrameTimeVector'):
            frame_time_vector = self.ds.FrameTimeVector._list
            # print(frame_time_vector)
            unique_elements, counts_elements = np.unique(frame_time_vector, return_counts=True)
            idx = np.argmax(counts_elements)
            frame_time = unique_elements[idx]

        fps = 1 / frame_time * 1000 if frame_time is not None else None

        return fps

    def get_box_ceus_bmode(self):
        if self.ds is None:
            self.read_header()

        box_ceus, box_bmode = None, None
        if hasattr(self.ds, 'SequenceOfUltrasoundRegions'):
            regions = self.ds.SequenceOfUltrasoundRegions._list
            box_ceus = regions[0].RegionLocationMinX0, regions[0].RegionLocationMinY0, regions[0].RegionLocationMaxX1, \
                       regions[0].RegionLocationMaxY1
            box_ceus = (box_ceus[0] + 1, box_ceus[1] + 1, box_ceus[2] - 1, box_ceus[3] - 1)

            if len(regions) > 1:
                # box_bmode = regions[1].RegionLocationMinX0, regions[1].RegionLocationMinY0, regions[1].RegionLocationMaxX1, \
                #             regions[1].RegionLocationMaxY1
                # box_bmode = (box_bmode[0] + 1, box_bmode[1] + 1, box_bmode[2] - 1, box_bmode[3] - 1)
                x1 = regions[1].RegionLocationMinX0 + 1
                box_bmode = (x1, box_ceus[1], x1 + box_ceus[2] - box_ceus[0], box_ceus[3])

        return box_ceus, box_bmode

    def get_pixel_spacing(self):
        delta_x, delta_y = None, None
        if hasattr(self.ds, 'SequenceOfUltrasoundRegions'):
            delta_x = self.ds.SequenceOfUltrasoundRegions._list[0].PhysicalDeltaX * 10
            delta_y = self.ds.SequenceOfUltrasoundRegions._list[0].PhysicalDeltaY * 10

        return (delta_x, delta_y)

    def get_loopend_datetime(self):
        ContentDate = self.ds.ContentDate
        ContentTime = self.ds.ContentTime
        content = ContentDate + ' ' + ContentTime
        if '.' in content:
            dt = datetime.strptime(content, '%Y%m%d %H%M%S.%f')
        else:
            dt = datetime.strptime(content, '%Y%m%d %H%M%S')
        return dt

    def read_data_gray(self):
        data = self.read_data()
        if data is None: return
        seq = np.array([cv.cvtColor(frame, cv.COLOR_BGR2GRAY) for frame in data])

        return seq

    def play(self, win_name=''):
        data = self.read_data()
        if data is None: return

        if self.NumberOfFrames <= 1:
            cv.imshow(win_name, data)
            cv.waitKey()
        else:
            for frame in data:
                cv.imshow(win_name, frame)
                cv.waitKey(40)

    def draw_play(self, boxes):
        data = self.read_data()
        if data is None: return

        for img in data:
            for box in boxes:
                left_top = box[:2]
                right_bottom = box[2:]
                cv.rectangle(img, left_top, right_bottom, color=(0, 255, 0), thickness=1)
            cv.imshow('', img)
            cv.waitKey(100)

    def to_pic(self, save_dir, extname='png'):
        data = self.read_data()
        if data is None: return

        # filename = self.dcm_file.split('/')[-1]
        # save_dir = os.path.join(save_dir, filename)
        # os.makedirs(save_dir, exist_ok=True)

        for idx, frame in enumerate(data):
            save_file = os.path.join(save_dir, '{0}.{1}'.format(idx, extname))
            cv.imwrite(save_file, frame)


    # def to_pic_or_video(self, save_dir):
    #     data = self.read_data()
    #     if data is None: return
    #
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #     filename = self.dcm_file.split('/')[-1]
    #
    #     if self.NumberOfFrames <= 1:
    #         save_file = os.path.join(save_dir, filename+'.png')
    #         cv.imwrite(save_file, data)
    #     else:
    #         save_file = os.path.join(save_dir, filename+'.avi')
    #         vwriter = None
    #         #prog_bar = mmcv.ProgressBar(len(data))
    #         for frame in data:
    #             if vwriter is None:
    #                 height, width = frame.shape[:2]
    #                 resolution = (width, height)
    #                 fourcc = 'XVID'
    #                 fps = 24
    #
    #                 vwriter = cv.VideoWriter(save_file, VideoWriter_fourcc(*fourcc), fps,
    #                                          resolution)
    #
    #             vwriter.write(frame)
    #
    #             #prog_bar.update()

    def to_npy(self, save_file):
        data = self.read_data()
        if data is None: return

        path = Path(save_file)
        save_dir = str(path.parent)
        os.makedirs(save_dir, exist_ok=True)

        save_file = save_file + '.npy'

        pickle.dump(data, open(save_file, 'wb'))


def load_ceus_bmode_bydcmfile(dcmfile):
    print('reading dicom...{0}'.format(dcmfile))
    dcm_reader = DicomReader(dcmfile)
    dcm_reader.read_header()
    # seq = dcm_reader.read_data_gray()
    seq = dcm_reader.read_data()
    print('data is loaded.')
    # show_seq(seq)

    return seq


if __name__ == '__main__':
    dcmfile = '/media/gdp/25793662-6b5a-431d-8402-87c5bd9357df/dataset/CEUS/TJ3CH-TBUsed/第1批/270153/DICOM/N1/5 CEUS/QC.US._.0001.0023.2017.09.08.11.44.53.973460.863557.IMA'
    # dcmfile = '/media/gdp/25793662-6b5a-431d-8402-87c5bd9357df/dataset/CEUS/TJ3CH-TBUsed/第1批/270153/DICOM/BG/2 Right intercostal section/QC.US._.0001.0019.2017.09.08.11.44.53.973460.877914.IMA'
    dcmreader = DicomReader(dcmfile).read_header()
    print(dcmreader.get_fps())
    print(dcmreader.get_pixel_spacing())
    print(dcmreader.get_box_ceus_bmode())
    print(dcmreader.get_framenum())
    print(dcmreader.get_loopend_datetime())

    save_dir = '/home/gdp/Pictures/270153'
    os.makedirs(save_dir, exist_ok=True)
    data = DicomReader(dcmfile).read_data()

    for i in [0, 400, 800, 1200]:
        frame = data[i]
        save_file = os.path.join(save_dir, f'{i}.png')
        cv.imwrite(save_file, frame)












