# -*- coding: utf-8 -*-
"""
:Authors: Qizhong Lin <qizhong.lin@philips.com>,
:Copyright: This file contains proprietary information of Philips 
            Innovative Technologies. Copying or reproduction without prior
            written approval is prohibited.

            Philips internal use only - no distribution outside Philips allowed
"""
import os.path
from tqdm import tqdm
from pprint import pprint
import random
import json
import shutil
from collections import defaultdict
from sklearn.model_selection import train_test_split

from classification.clean_data.dicom.DicomReader import DicomReader
from classification.config import random_seed
from read_echo import mat2pic
from split_raw_data import DATA_ROOT, RAW_ROOT, split_train_val_test

MAT_ROOT = os.path.join(DATA_ROOT, 'mat')


def split_train_val_test_raw(raw_info_file=os.path.join(RAW_ROOT, 'split_info.json')):
    raw_info = json.loads(open(raw_info_file).read())

    files = {}
    for mode, value in raw_info["dcm file"].items():
        cls_files = {}
        for cls, dcmfiles in value.items():
            dcmfiles = [file[1] for file in dcmfiles]
            matfiles = []
            for dcmfile in dcmfiles:
                matfile = dcmfile.replace(RAW_ROOT, str(os.path.join(MAT_ROOT, cls)))
                matfile = matfile.replace("DICOM/", "")
                matfile += ".mat"
                # assert os.path.exists(matfile)
                matfiles.append(matfile)

            cls_files[cls] = matfiles
        files[mode] = cls_files

    return files, raw_info


def split_patient_Philips(MAT_CCA, test_size=0.25, val_size=0.25):
    patients = {}
    for name in os.listdir(MAT_CCA):
        if 'Philips' in name:
            patient_dir = os.path.join(MAT_CCA, name)
            patients[name] = [os.path.join(patient_dir, file) for file in os.listdir(patient_dir)]
    patient_train, patient_val, patient_test = split_train_val_test(patients, test_size=test_size, val_size=val_size)

    return patient_train, patient_val, patient_test



def main():
    # get train/val/test from raw
    files, raw_info = split_train_val_test_raw()

    # add train/val/test from mat
    for cls in ['CCA', 'VERT']:
        patient_train, patient_val, patient_test = split_patient_Philips(os.path.join(MAT_ROOT, cls))
        files['train'][cls].extend(patient_train)
        files['val'][cls].extend(patient_val)
        files['test'][cls].extend(patient_test)

    # add patients
    raw_info["patients"]["train"].extend(patient_train)
    raw_info["patients"]["val"].extend(patient_val)
    raw_info["patients"]["test"].extend(patient_test)
    data = {
        'files': files,
        'patients': {
            'train': list(raw_info["patients"]["train"]),
            'val': list(raw_info["patients"]["val"]),
            'test': list(raw_info["patients"]["test"])
        }
    }
    json_fie = os.path.join(MAT_ROOT, 'split_info.json')
    with open(json_fie, 'w') as ftxt:
        json.dump(data, ftxt, indent=4)

    # mat to png
    PIC_DIR = os.path.join(DATA_ROOT, "PIC")
    for mode, value in data["files"].items():
        for cls, matfiles in value.items():
            cls_dir = os.path.join(PIC_DIR, cls)
            os.makedirs(cls_dir, exist_ok=True)
            print(f"process {mode}, {cls} ...")
            for matfile in tqdm(matfiles):
                _, basename = os.path.split(matfile)
                save_dir = os.path.join(cls_dir, basename[:-4])
                mat2pic(matfile, save_dir)










if __name__ == '__main__':
    main()
