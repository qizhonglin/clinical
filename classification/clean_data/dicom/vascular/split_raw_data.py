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

DATA_ROOT = os.path.join("/media/qzlin/25793662-6b5a-431d-8402-87c5bd9357df/dataset/Hackthon/vascular/")
RAW_ROOT = os.path.join(DATA_ROOT, "raw")

issues = [
    "ColorAutoscaleFP_ScanningSession_20230802/DICOM/IM_0812",
    "ColorAutoscaleFP_ScanningSession_20230802/DICOM/IM_0390",
    "ColorAutoscaleFP_ScanningSession_20230802/DICOM/IM_0814",
    "ColorAutoscaleFP_ScanningSession_20230802/DICOM/IM_0392",
    "ColorAutoscaleFP_ScanningSession_20230802/DICOM/IM_0593",
    "ColorAutoscaleFP_ScanningSession_20230802/DICOM/IM_0391"
]


def collect_infos(DATA_ROOT):
    infos = {}

    for group in os.listdir(DATA_ROOT):
        group_dir = os.path.join(DATA_ROOT, group)
        if not os.path.isdir(group_dir): continue

        print(f"processing {group_dir}...")

        dcm_dir = os.path.join(group_dir, "DICOM")
        for name in tqdm(os.listdir(dcm_dir)):
            dcm_file = os.path.join(dcm_dir, name)
            dcmreader = DicomReader(dcm_file).read_header()
            try:
                info = {
                    'PatientName': (group, dcmreader.PatientName),
                    'PatientID': dcmreader.PatientID,
                    'Manufacturer': dcmreader.Manufacturer,
                    'fps': dcmreader.get_fps(),
                    'pixel spacing': dcmreader.get_pixel_spacing(),
                    'frame num': dcmreader.get_framenum(),
                    'View': dcmreader.ds[0x00082127].value
                }
                # print(info)
                infos[dcm_file] = info
            except:
                print(f"issues happen: {dcm_file}")

    return infos


def organize_infos(infos, is_CAR=True):
    patient_info = defaultdict(list)
    for file, info in infos.items():
        patient_name = tuple(info["PatientName"])
        patient_info[patient_name].append({
            "file": file,
            **info
        })

    if is_CAR:
        car_info = {patient_name: info for patient_name, info in patient_info.items() if "CAR" in patient_name[1]}
    else:
        car_info = {patient_name: info for patient_name, info in patient_info.items() if "CAR" not in patient_name[1]}

    patient_car_info = defaultdict(list)
    for patient_name, info in car_info.items():
        pid = patient_name[1].split(' ')[0]
        car_on_off_info = defaultdict(list)
        for ele in info:
            car_on_off_info[ele["PatientName"][1]].append(ele)
            del ele["PatientName"]
        patient_car_info[(patient_name[0], pid)].append(dict(car_on_off_info))

    return dict(patient_car_info)


def get_dcms(patient_car_info):
    verts = []
    CCAs = []
    for patient_name, info in patient_car_info.items():
        for type_info in info:
            values = list(type_info.values())[0]
            vert = [(val["View"], val["file"]) for val in values if "VERT" in val["View"]]
            CCA = [(val["View"], val["file"]) for val in values if "VERT" not in val["View"]]

            verts.extend(vert)
            CCAs.extend(CCA)

    return verts, CCAs


def split_train_val_test(patients, test_size=0.25, val_size=0.25):
    patient_train, patient_test = train_test_split(list(patients), test_size=test_size, random_state=random_seed,
                                                   shuffle=True)
    patient_train = {id: patients[id] for id in patient_train}
    patient_test = {id: patients[id] for id in patient_test}

    patient_val = {}
    if val_size > 0:
        patient_train, patient_val = train_test_split(list(patient_train), test_size=val_size, random_state=random_seed,
                                                      shuffle=True)
        patient_train = {id: patients[id] for id in patient_train}
        patient_val = {id: patients[id] for id in patient_val}

    return patient_train, patient_val, patient_test


def dcm2pic(verts_val, RAW_ROOT, PIC_DIR):
    for car_type, dcm_file in tqdm(verts_val):
        name = dcm_file.replace(RAW_ROOT, "")
        name = name.replace("/", "_")
        save_dir = os.path.join(PIC_DIR, name)
        os.makedirs(save_dir, exist_ok=True)
        DicomReader(dcm_file).to_pic(save_dir=save_dir)


def split_patient_car(infos):
    patients = organize_infos(infos)
    # print(patients)

    patient_train, patient_val, patient_test = split_train_val_test(patients, val_size=0.25)

    verts_train, CCAs_train = get_dcms(patient_train)
    verts_val, CCAs_val = get_dcms(patient_val)
    verts_test, CCAs_test = get_dcms(patient_test)

    CCAs_train = [(mode, file) for mode, file in CCAs_train if "OPT" in mode]
    CCAs_val = [(mode, file) for mode, file in CCAs_val if "OPT" in mode]
    CCAs_test = [(mode, file) for mode, file in CCAs_test if "OPT" in mode]
    verts_val = [(mode, file) for mode, file in verts_val if "OPT" in mode]
    verts_test = [(mode, file) for mode, file in verts_test if "OPT" in mode]

    return (patient_train, patient_val, patient_test), (CCAs_train, CCAs_val, CCAs_test), (
    verts_train, verts_val, verts_test)


def split_patient_no_car(infos, patient_val, patient_test, CCAs_train, CCAs_val, CCAs_test):
    patients_no_car = organize_infos(infos, is_CAR=False)
    patients_no_car_val = set(patients_no_car).intersection(set(patient_val))
    patients_no_car_test = set(patients_no_car).intersection(set(patient_test))
    patients_no_car_train = set(patients_no_car) - set(patient_val) - set(patients_no_car_test)
    patients_no_car_train = {id: patients_no_car[id] for id in patients_no_car_train}
    patients_no_car_val = {id: patients_no_car[id] for id in patients_no_car_val}
    patients_no_car_test = {id: patients_no_car[id] for id in patients_no_car_test}
    no_cars_train = get_backgrounds(patients_no_car_train)
    no_cars_val = get_backgrounds(patients_no_car_val)
    no_cars_test = get_backgrounds(patients_no_car_test)
    no_cars_train = sample_background(no_cars_train, train_num=len(CCAs_train))
    no_cars_val = sample_background(no_cars_val, train_num=len(CCAs_val))
    no_cars_test = sample_background(no_cars_test, train_num=len(CCAs_test))

    return no_cars_train, no_cars_val, no_cars_test


def get_backgrounds(patients_no_car):
    no_cars = []
    for patient_name, info in patients_no_car.items():
        for type_info in info:
            values = list(type_info.values())[0]
            no_car = [(val["View"], val["file"]) for val in values]
            no_cars.extend(no_car)

    no_cars = [(mode, file) for mode, file in no_cars if "OPT" in mode]

    return no_cars


def sample_background(no_cars, train_num):
    no_cars_dict = defaultdict(list)
    for name, file in no_cars:
        no_cars_dict[name].append(file)

    sample_num = int(train_num / len(no_cars_dict)) + 1

    no_cars = []
    for key, value in no_cars_dict.items():
        sample = value
        if len(value) > sample_num:
            sample = random.sample(no_cars_dict[key], sample_num)
        no_cars.extend([(key, item) for item in sample])

    return no_cars


def copy_data_for_annotation(CCAs_train, CCAs_val, CCAs_test,
                             verts_train, verts_val, verts_test,
                             RAW_ROOT, dst_dir):
    def _copy_data(verts_val, RAW_ROOT, vert):
        for mode, file in verts_val:
            file_dst = file.replace(RAW_ROOT, vert)
            parent_dir = os.path.dirname(file_dst)
            os.makedirs(parent_dir, exist_ok=True)
            shutil.copy(file, file_dst)

    vert = os.path.join(dst_dir, 'vert')
    CCA = os.path.join(dst_dir, 'CCA')
    _copy_data(verts_val, RAW_ROOT, vert)
    _copy_data(verts_test, RAW_ROOT, vert)
    _copy_data(verts_train, RAW_ROOT, vert)
    _copy_data(CCAs_val, RAW_ROOT, CCA)
    _copy_data(CCAs_test, RAW_ROOT, CCA)
    _copy_data(CCAs_train, RAW_ROOT, CCA)


def display_info(data):
    info = {
        "loop num": {
            "train": {
                "VERT": len(data["train"]["VERT"]),
                "CCA": len(data["train"]["CCA"]),
                "NO CAR": len(data["train"]["NO CAR"]),
            },
            "val": {
                "VERT": len(data["val"]["VERT"]),
                "CCA": len(data["val"]["CCA"]),
                "NO CAR": len(data["val"]["NO CAR"]),
            },
            "test": {
                "VERT": len(data["test"]["VERT"]),
                "CCA": len(data["test"]["CCA"]),
                "NO CAR": len(data["test"]["NO CAR"]),
            },
        },
        "dcm file": {
            "train": {
                "VERT": data["train"]["VERT"],
                "CCA": data["train"]["CCA"],
                "NO CAR": data["train"]["NO CAR"],
            },
            "val": {
                "VERT": data["val"]["VERT"],
                "CCA": data["val"]["CCA"],
                "NO CAR": data["val"]["NO CAR"],
            },
            "test": {
                "VERT": data["test"]["VERT"],
                "CCA": data["test"]["CCA"],
                "NO CAR": data["test"]["NO CAR"],
            },
        }
    }
    print(info)
    return info


def main():
    json_fie = os.path.join(RAW_ROOT, 'dcm_info.json')

    # infos = collect_infos(RAW_ROOT)
    # print(infos)
    # with open(json_fie, 'w') as ftxt:
    #     json.dump(infos, ftxt, indent=4)

    infos = json.loads(open(json_fie).read())
    (patient_train, patient_val, patient_test), (CCAs_train, CCAs_val, CCAs_test), (
    verts_train, verts_val, verts_test) = split_patient_car(infos)
    no_cars_train, no_cars_val, no_cars_test = split_patient_no_car(infos, patient_val, patient_test, CCAs_train,
                                                                    CCAs_val, CCAs_test)

    no_cars = no_cars_train + no_cars_val + no_cars_test
    cars = CCAs_train + CCAs_val + CCAs_test + verts_train + verts_val + verts_test
    empty = set(no_cars).intersection(set(cars))
    assert not empty

    data = {
        "train": {
            "VERT": verts_train,
            "CCA": CCAs_train,
            "NO CAR": no_cars_train
        },
        "val": {
            "VERT": verts_val,
            "CCA": CCAs_val,
            "NO CAR": no_cars_val
        },
        "test": {
            "VERT": verts_test,
            "CCA": CCAs_test,
            "NO CAR": no_cars_test
        },
    }
    info = display_info(data)
    info.update({
        'patients': {
            'train': list(patient_train),
            'val': list(patient_val),
            'test': list(patient_test)
        }
    })

    json_fie = os.path.join(RAW_ROOT, 'split_info.json')
    with open(json_fie, 'w') as ftxt:
        json.dump(info, ftxt, indent=4)



if __name__ == '__main__':
    main()
