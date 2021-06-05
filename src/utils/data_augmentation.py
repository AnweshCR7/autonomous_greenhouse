import os
import cv2
import glob
import random
import numpy as np
import matplotlib.pyplot as plt


def plot_image(img):
    plt.axis("off")
    plt.imshow(img, origin='upper')
    plt.show()


def flip(img, dir_flag):
    # if flag:
    return cv2.flip(img, dir_flag)
    # else:
    #     return img


def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def augment(dataset_dir, augmentations, save_copy=False):
    image_paths = glob.glob(f"{dataset_dir}/RGB_*.png")

    # Lets apply the augmentation to each image!
    for image_path in image_paths[:1]:
        # Will be useful for saving
        img_filename = image_path.split('/')[-1].split('.')[0]
        img = cv2.imread(image_path)

        # Horizontal Flip -> flip(img, True)
        img_hf = flip(img, 1)
        if save_copy:
            cv2.imwrite(f"{dataset_dir}/hf_{img_filename}", img_hf)

        # Maybe we apply vertical flip to the horizontally flipped images as well?
        # For now hard coding that process i.e. VF(HF(img)) by saving two copies
        # Vertical Flip -> flip(img, False)
        img_vf = flip(img, 0)
        if save_copy:
            cv2.imwrite(f"{dataset_dir}/vf_{img_filename}", img_vf)
            # apply vertical flip to a horizontally flipped image.
            cv2.imwrite(f"{dataset_dir}/vfhf_{img_filename}", flip(img_hf, 0))

        # # Horizontal Flip
        # img_hf = horizontal_flip(img, True)
        # if save_copy:
        #     cv2.imwrite(f"{dataset_dir}/hf_{img_filename}", img_hf)


if __name__ == '__main__':
    dataset_dir = "../../data/FirstTrainingData_AUG"
    augmentations = ["horizontal_flip"]
    # expect to increase the dataset 4x
    augment(dataset_dir, augmentations, save_copy=False)