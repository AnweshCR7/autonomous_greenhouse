import albumentations as A
import torch
import numpy as np
import cv2
import config
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from torchvision import transforms
from label import Label
# from PIL import Image
# from PIL import ImageFile


# ImageFile.LOAD_TRUNCATED_IMAGES = True
def plot_image(img):
    plt.axis("off")
    plt.imshow(img)
    plt.show()


class DataLoaderSegmentation:
    def __init__(self, img_paths, mask_paths, metadata, center_crop=None, resize=None, predict=False, classes=None, preprocessing=None, augmentation=None):

        self.img_paths = img_paths
        self.mask_paths = mask_paths
        # self.targets_list = metadata
        self.predict = predict
        # if not self.predict:
        self.metadata_df = pd.read_csv(metadata)
        # else:
        #     self.targets_df = None
        self.center_crop = center_crop
        self.resize = resize
        # self.scalerfile = config.SCALERFILE
        self.scaler = pickle.load(open(f"{config.SCALER_PATH}{config.SCALERFILE}", 'rb'))
        # convert str names to class values on masks
        self.class_values = [cls for cls in Label]

        self.augmentation = augmentation

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        # self.preprocess = transforms.Compose([
        #     # transforms.ToTensor(),
        #     transforms.Normalize(mean=mean, std=std),
        #     # transforms.Resize((resize[1], resize[0]))
        # ])

        # # This is more like in-place augmentation
        # self.augmentation_pipeline = augmentation

        self.augmentation_pipeline =  A.Compose(
            [
                # A.CenterCrop(self.center_crop, self.center_crop),
                # A.HorizontalFlip(p=0.5),
                A.Resize(224, 224)
                # A.Normalize(
                #     mean, std, max_pixel_value=1.0, always_apply=True
                # )
            ]
        )
        self.preprocessing = preprocessing
        # A.Compose(
        #     [
        #         # A.CenterCrop(self.center_crop, self.center_crop),
        #         # A.HorizontalFlip(p=0.5),
        #         # A.Resize(resize[1], resize[0])
        #         A.Normalize(
        #             mean, std, max_pixel_value=1.0, always_apply=True
        #         )
        #     ]
        # )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # Image has 4 channels -> converting to RGB
        image = cv2.imread(self.img_paths[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # scale between 0,1
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Get the image number
        image_num = self.img_paths[index].split('/')[-1].split('.')[0].split('_')[-1]

        # convert to numpy array
        image = np.array(image).astype('uint8')
        # Convert to form: CxHxW

        # Apply image transforms
        # image = self.preprocess(image)

        mask = cv2.imread(self.mask_paths[index], 0)

        # Separate out the masks for each class and stack
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.augmentation_pipeline:
            augmented_sample = self.augmentation_pipeline(image=image, mask=mask)
            image, mask = augmented_sample['image'], augmented_sample['mask']

        # # apply augmentations
        # if self.augmentation:
        #     sample = self.augmentation(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # image = np.transpose(image, (2, 0, 1))

        targets = "plant"
        # targets = self.metadata_df[self.metadata_df["ImageName"] == f"Image{image_num}"]
        # target_class = targets["Variety"].values[0]


        # if self.resize is not None:
        #     # write as HxW
        #     image = cv2.resize(image, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_CUBIC)

        if self.predict:
            return torch.tensor(image, dtype=torch.float)

        '''
        This will take care of all augmentations that stem from the original image name as long as we prefix names in the augmented counterparts.
        Reason: the image number is the same for all augmentations.
        '''
        # # WHEN USING .csv FILE
        # targets = self.targets_df[self.targets_df["ImageName"] == f"Image{image_num}"]
        # if targets.shape[0] == 0:
        #     print(f"Image{image_num}")
        # targets = targets[config.FEATURES].values
        # targets = self.scaler.transform(targets).flatten()
        # for feature in config.FEATURES:
        #     targets.append(target_dict[feature])

        # WHEN USING .json FILE
        # target_dict = self.targets_list[f"Image{image_num}"]
        # targets = []
        # for feature in config.FEATURES:
        #     targets.append(target_dict[feature])

        return image, mask
        # return {
        #     "image": torch.tensor(image, dtype=torch.float),
        #     "mask": torch.tensor(mask, dtype=torch.float),
        #     "target_class": target_class
        # }