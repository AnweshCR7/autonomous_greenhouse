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
from torchvision import transforms

# ImageFile.LOAD_TRUNCATED_IMAGES = True
def plot_image(img):
    plt.axis("off")
    plt.imshow(img, origin='upper')
    plt.show()


class DataLoaderLettuceNet:
    def __init__(self, img_paths, metadata, center_crop=None, resize=None, predict=False, add_features=None, augmentations=None):

        self.img_paths = img_paths
        # if segmentation_paths is not None:
        #     self.segmentation_paths = segmentation_paths
        # self.targets_list = metadata
        self.predict = predict
        if not self.predict:
            self.targets_df = pd.read_csv(metadata)
        else:
            self.targets_df = None
        self.center_crop = center_crop
        self.resize = resize
        # self.scalerfile = config.SCALERFILE
        self.scaler = pickle.load(open(f"{config.SCALER_PATH}{config.SCALERFILE}", 'rb'))
        self.scaler_x = pickle.load(open(f"{config.SCALER_PATH}{config.SCALERFILE_X}", 'rb'))
        self.scaler_y = pickle.load(open(f"{config.SCALER_PATH}{config.SCALERFILE_Y}", 'rb'))

        if add_features:
            self.add_features_df = pd.read_csv(add_features)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        # Maybe add more augmentations
        # transforms = transforms.Compose[
        #     LongestMaxSize(max_size=500),
        #     HorizontalFlip(p=0.5),
        #     PadIfNeeded(500, 600, border_mode=0, value=0),
        #     JpegCompression(quality_lower=70, quality_upper=100, p=1),
        #     RandomBrightnessContrast(0.3, 0.3),
        #     Cutout(max_h_size=32, max_w_size=32, p=1)
        # ]

        # List of Augmentations:
        # Centre crop
        # resize
        # addhueandcolorsaturation
        # linearcontrast
        # affinesclae
        # affinerotate
        # affinesheer


        # This is more like in-place augmentation
        if augmentations == "train":
            self.augmentation_pipeline = A.Compose(
                [
                    A.CenterCrop(height= self.center_crop[1], width=self.center_crop[0]),
                    # A.PadIfNeeded(min_height=1080, min_width=1920, p=1),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Resize(resize[1], resize[0]),
                    # A.RandomRotate90(p=1),
                    # A.RandomScale(0.25),
                    # A.RandomBrightnessContrast(p=0.8),
                    # A.IAAAffine(rotate=0.5, p=0.8),
                    # A.IAAAffine(shear=0.5, p=0.8),

                    # A.Normalize(
                    #     mean, std, max_pixel_value=255.0, always_apply=True
                    # )
                ]
            )
        else:
            if augmentations == "train":
                self.augmentation_pipeline = A.Compose(
                    [
                        A.CenterCrop(height=self.center_crop[1], width=self.center_crop[0]),
                        A.Resize(resize[1], resize[0]),
                    ]
                )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # Image has 4 channels -> converting to RGB
        image = cv2.imread(self.img_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Get the image number
        image_num = self.img_paths[index].split('/')[-1].split('.')[0].split('_')[-1]
        seg_mask = cv2.imread(f"{config.SEG_DIR}/Seg_{image_num}.png")
        seg_mask = seg_mask[:, :, 0]
        # Find class
        cultivar = np.unique(seg_mask)[1]
        # Get segmentation of class
        seg_mask_lettuce = (seg_mask == cultivar).astype('uint8')
        # expand single dim
        seg_mask_lettuce = np.expand_dims(seg_mask_lettuce, axis=2)
        segmented_image = cv2.bitwise_or(image, image, mask=seg_mask_lettuce)

        # convert to numpy array
        image = np.array(segmented_image)

        # color to fill
        color = np.array([0, 255, 0], dtype='uint8')

        # # equal color where mask, else image
        # # this would paint your object silhouette entirely with `color`
        # masked_img = np.where(seg_mask[...], color, image)
        #
        # # use `addWeighted` to blend the two images
        # # the object will be tinted toward `color`
        # out = cv2.addWeighted(seg_mask, 0.8, masked_img, 0.2, 0)


        # Apply image transforms
        image = self.augmentation_pipeline(image=image)['image']
        # if self.resize is not None:
        #     # write as HxW
        #     image = cv2.resize(image, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_CUBIC)

        # Convert to form: CxHxW
        image = np.transpose(image, (2,0,1)).astype(np.float32)

        if self.predict:
            return torch.tensor(image, dtype=torch.float)

        '''
        This will take care of all augmentations that stem from the original image name as long as we prefix names in the augmented counterparts.
        Reason: the image number is the same for all augmentations.
        '''
        # WHEN USING .csv FILE
        targets = self.targets_df[self.targets_df["ImageName"] == f"Image{image_num}"]
        if targets.shape[0] == 0:
            print(f"Image{image_num}")
        targets = targets[config.FEATURES].values
        targets = self.scaler_y.transform(targets).flatten()

        # Lets work with add_features
        features = self.add_features_df[self.add_features_df["Unnamed: 0"] == f"Image{image_num}"]

        # features['Shuffle'] = features['Area'] * features['Height'] * features['Volume'] * features[
        #     'Diameter']
        # features['Shuffle'] = np.log(features['Shuffle'])
        # extra_features = features['Shuffle'].values


        features = features[config.ADD_FEATURES].values
        features = self.scaler_x.transform(features).flatten()
        # np.append(features, extra_features)



        # for feature in config.FEATURES:
        #     targets.append(target_dict[feature])

        # WHEN USING .json FILE
        # target_dict = self.targets_list[f"Image{image_num}"]
        # targets = []
        # for feature in config.FEATURES:
        #     targets.append(target_dict[feature])

        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float)
        }