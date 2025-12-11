# -*- coding: utf-8 -*-
"""
:Authors: Qizhong Lin <qizhong.lin@philips.com>,
:Copyright: This file contains proprietary information of Philips 
            Innovative Technologies. Copying or reproduction without prior
            written approval is prohibited.

            Philips internal use only - no distribution outside Philips allowed
"""
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
import json
from tqdm import tqdm

import torch
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image
)

from LymphNode.classification.dl_based.Model2D import get_model
from utils.util4torch import get_device, net2device
from LymphNode.config import CHECKPOINT_DIR, classes, DATA_ROOT
from LymphNode.classification.dl_based.dataset import preprocess_image

from LymphNode.classification.dl_based.dataset import (
    DATA_DIR,
    EXTERNAL_DATA_DIR,
    get_dataset_train_val_test,
    get_dataset_external_test,
    get_dataset_train_val_test_ext,
    split_train_val_test,
    get_data,
    split_train_val_test_ext)
from LymphNode.classification.dl_based.solver import infer_each_class


def load_model():
    model_file = os.path.join(CHECKPOINT_DIR, 'normal_train_without_external_data', 'best.pth')

    config = {
        "backbone": 'resnet18',
        "fix_depth": 1,
        "drop_out": 0.5,
        "hidden_dim": 0
    }
    net = get_model(config["fix_depth"], config["backbone"], len(classes), config["drop_out"], config["hidden_dim"])
    net2device(net)

    net.load_state_dict(torch.load(model_file))

    return net


def _infer_image(net, input_tensor):
    with torch.no_grad():
        outputs = net(input_tensor)
        # preds = torch.softmax(outputs, 1)
        _, predictions = torch.max(outputs, 1)
        label_idx = predictions.data.cpu().numpy()[0]
        label = classes[label_idx]

    return label


def get_cam(model, input_tensor, rgb_img, cam_algorithm=GradCAM):
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    target_layers = [model.layer4]

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [ClassifierOutputTarget(281)]
    # targets = [ClassifierOutputTarget(281)]
    targets = None

    with cam_algorithm(model=model,
                       target_layers=target_layers) as cam:
        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=False,   # Apply test time augmentation to smooth the CAM
                            eigen_smooth=False  # Reduce noise by taking the first principle component of cam_weights*activations
                            )

        grayscale_cam = grayscale_cam[0, :]

        rgb_img_norm = rgb_img.resize(grayscale_cam.shape)
        rgb_img_norm = np.float32(rgb_img_norm) / 255
        cam_image = show_cam_on_image(rgb_img_norm, grayscale_cam, use_rgb=True, image_weight=0.8)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, device=get_device())
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cam_image = cv2.resize(cam_image, rgb_img.size)
    cam_gb = cv2.resize(cam_gb, rgb_img.size)
    gb = cv2.resize(gb, rgb_img.size)

    return cam_image, cam_gb, gb


def cam_images(
        imagefiles,
        output_dir=os.path.join(DATA_ROOT, 'cache')
):
    model = load_model()
    model.eval()

    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise,
        'kpcacam': KPCA_CAM
    }
    # for method in tqdm(methods):
    method = 'gradcam'

    for img_file in tqdm(imagefiles):
        rgb_img = Image.open(img_file)
        input_tensor = preprocess_image(rgb_img).to(get_device())

        # label = _infer_image(model, input_tensor)
        # print(label)

        cam_image, cam_gb, gb = get_cam(model, input_tensor, rgb_img, methods[method])

        method_dir = os.path.join(output_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        name = Path(img_file).name[:-4]
        cam_output_path = os.path.join(method_dir, f'{name}_cam.jpg')
        gb_output_path = os.path.join(method_dir, f'{name}_gb.jpg')
        cam_gb_output_path = os.path.join(method_dir, f'{name}_cam_gb.jpg')

        shutil.copy(img_file, os.path.join(method_dir, Path(img_file).name))
        cv2.imwrite(cam_output_path, cam_image)
        cv2.imwrite(gb_output_path, gb)
        cv2.imwrite(cam_gb_output_path, cam_gb)


def _cam_test(model, test_ds, output_dir=os.path.join(DATA_ROOT, 'cache/test_ds')):
    truth, probs = infer_each_class(model, test_ds, classes)
    preds = np.argmax(probs, axis=1)
    acc = sum(truth == preds) / len(truth)
    print(f'{Path(output_dir).name} acc: {acc}')
    mask = truth == preds
    images = np.asarray(test_ds.images)
    images_correct = images[mask]
    images_wrong = images[~mask]

    cam_images(images_correct.tolist(),  os.path.join(output_dir, 'correct'))
    cam_images(images_wrong.tolist(), os.path.join(output_dir, 'wrong'))


def cam_test():
    data = split_train_val_test(DATA_DIR)
    train_ds, val_ds, test_int_ds = get_dataset_train_val_test(data)
    test_ext_ds = get_dataset_external_test(*get_data(EXTERNAL_DATA_DIR))

    model = load_model()

    _cam_test(model, test_int_ds, output_dir=os.path.join(DATA_ROOT, 'cache/test_int_ds'))
    _cam_test(model, test_ext_ds, output_dir=os.path.join(DATA_ROOT, 'cache/test_ext_ds'))


def roi2whole():
    save_file = os.path.join(DATA_ROOT, 'cache', 'roi_box.json')
    roi_box = json.loads(open(save_file).read())
    file_roiboxs = {}
    for group in roi_box:
        for mode in roi_box[group]:
            for file in roi_box[group][mode]:
                file_roiboxs[file] = roi_box[group][mode][file]

    file_imagepaths = {}
    for group in ['huaxi', 'tianfu']:
        images_dir = os.path.join(DATA_ROOT, group, "images")
        for mode in ["malignant", "benign"]:
            mode_images_dir = os.path.join(images_dir, mode)
            for file in tqdm(os.listdir(mode_images_dir)):
                file_imagepaths[file] = os.path.join(mode_images_dir, file)

    for group in ['test_ext_ds', 'test_int_ds']:
        group_dir = os.path.join(DATA_ROOT, 'cache', group)
        for result in os.listdir(group_dir):
            result_dir = os.path.join(group_dir, result, 'gradcam')
            imagefiles = [file for file in os.listdir(result_dir) if file.endswith('_cam.jpg')]

            for camfile in tqdm(imagefiles):
                file = camfile.split('_')[0] + '.jpg'
                imagefile = file_imagepaths[file]
                box = file_roiboxs[file]
                roifile = os.path.join(result_dir, camfile)

                x_s, y_s, x_e, y_e = box

                image = cv2.imread(imagefile)
                roi = cv2.imread(roifile)
                image[y_s:y_e, x_s:x_e, :] = roi

                dst_dir = os.path.join(group_dir, result, 'gradcam_whole')
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy(imagefile, os.path.join(dst_dir, file))
                cv2.imwrite(os.path.join(dst_dir, camfile), image)



def main():
    img_dir = '/media/qzlin/25793662-6b5a-431d-8402-87c5bd9357df1/dataset/LymphNode/huaxi/roi'
    imagefiles = [os.path.join(img_dir, file) for file in ['benign/3519326.jpg', 'malignant/4377836.jpg']]
    # cam_images(imagefiles)

    # cam_test()

    roi2whole()





if __name__ == '__main__':
    main()
