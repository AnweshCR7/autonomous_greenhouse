import albumentations as A
import torch
import numpy as np
import cv2
import config
import matplotlib.pyplot as plt
import pandas as pd
import pickle
# from PIL import Image
# from PIL import ImageFile

# ImageFile.LOAD_TRUNCATED_IMAGES = True
def plot_image(img):
    plt.axis("off")
    plt.imshow(img, origin='upper')
    plt.show()


class DataLoaderLettuceNet:
    def __init__(self, img_paths, metadata, center_crop=None, resize=None):

        self.img_paths = img_paths
        # self.targets_list = metadata
        self.targets_df = pd.read_csv(metadata)
        self.center_crop = center_crop
        self.resize = resize
        # self.scalerfile = config.SCALERFILE
        self.scaler = pickle.load(open(f"{config.SCALER_PATH}{config.SCALERFILE}", 'rb'))

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        # Maybe add more augmentations
        # transforms = [
        #     LongestMaxSize(max_size=500),
        #     HorizontalFlip(p=0.5),
        #     PadIfNeeded(500, 600, border_mode=0, value=0),
        #     JpegCompression(quality_lower=70, quality_upper=100, p=1),
        #     RandomBrightnessContrast(0.3, 0.3),
        #     Cutout(max_h_size=32, max_w_size=32, p=1)
        # ]

        # This is more like in-place augmentation
        self.augmentation_pipeline = A.Compose(
            [
                A.CenterCrop(self.center_crop, self.center_crop),
                A.HorizontalFlip(p=0.5),
                A.Resize(resize[1], resize[0])
                # A.Normalize(
                #     mean, std, max_pixel_value=255.0, always_apply=True
                # )
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # Image has 4 channels -> converting to RGB
        image = cv2.imread(self.img_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # Get the image number
        image_num = self.img_paths[index].split('/')[-1].split('.')[0].split('_')[-1]
        '''
        This will take care of all augmentations that stem from the original image name as long as we prefix names in the augmented counterparts.
        Reason: the image number is the same for all augmentations.
        '''
        # WHEN USING .csv FILE
        targets = self.targets_df[self.targets_df["ImageName"] == f"Image{image_num}"]
        if targets.shape[0] == 0:
            print(f"Image{image_num}")
        targets = targets[config.FEATURES].values
        targets = self.scaler.transform(targets).flatten()
        # for feature in config.FEATURES:
        #     targets.append(target_dict[feature])

        # WHEN USING .json FILE
        # target_dict = self.targets_list[f"Image{image_num}"]
        # targets = []
        # for feature in config.FEATURES:
        #     targets.append(target_dict[feature])

        # convert to numpy array
        image = np.array(image)
        # Apply image transforms
        image = self.augmentation_pipeline(image=image)['image']
        # if self.resize is not None:
        #     # write as HxW
        #     image = cv2.resize(image, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_CUBIC)

        # Convert to form: CxHxW
        image = np.transpose(image, (2,0,1)).astype(np.float32)

        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.float)
        }