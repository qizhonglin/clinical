
import cv2
import numpy as np
import os

print(cv2.__version__)


def get_green(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)  # 将图像转换为Lab颜色空间

    green_channel = lab[:, :, 1]            # 提取颜色通道（此处以绿色为例）

    # 根据阈值将像素点设为白色或黑色
    threshold = 120
    mask = 255 - cv2.threshold(green_channel, threshold, 255, cv2.THRESH_BINARY)[1]

    return mask


def edge2binary(edge):
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    result = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        result.append((w * h, c))
    result.sort(key=lambda x: x[0])

    bin = np.zeros_like(edge)
    cv2.fillConvexPoly(bin, result[-1][-1], 255)

    return bin


def cvt_contour_mask(image_file, mask_file):
    img = cv2.imread(image_file)

    mask = get_green(img)
    bin = edge2binary(mask)

    cv2.imwrite(mask_file, bin)


def cvt_mask_contour(image_file, mask_file, contour_file):
    img = cv2.imread(image_file)
    mask = cv2.imread(mask_file)

    erosion_size = 5
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))

    mask = cv2.erode(mask, element)

    edge = cv2.Canny(mask, 25, 255)
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

    cv2.imwrite(contour_file, img)

    return img



def main():
    from src.config import DATA_ROOT as root

    for group in ['huaxi', 'tianfu']:
        dir_images = os.path.join(root, group, "images")
        dir_masks = os.path.join(root, group, "masks")

        for name in ["malignant", "benign"]:
            print('processing {0}/{1}...'.format(group, name))

            dir_images_mode = os.path.join(dir_images, name+"_contour")
            dir_masks_mode = os.path.join(dir_masks, name)
            os.makedirs(dir_masks_mode, exist_ok=True)

            for file in os.listdir(dir_images_mode):
                image_file = os.path.join(dir_images_mode, file)
                mask_file = os.path.join(dir_masks_mode, file)
                # cvt_contour_mask(image_file, mask_file)

                # validate convert contour to mask is correct or not?
                dir_contour_mode = os.path.join(root, group, "contours", name)
                os.makedirs(dir_contour_mode, exist_ok=True)
                contour_file = os.path.join(dir_contour_mode, file)
                cvt_mask_contour(image_file, mask_file, contour_file)



    # cv2.imshow("img", img)
    # cv2.imshow("mask", mask)
    # cv2.imshow("bin", bin)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()