import os
import glob
import torch
import numpy as np
from pprint import pprint
from torchvision import datasets, transforms
import torch.nn as nn
from sklearn import model_selection
# from sklearn import metrics
import engine
import engine_segmentation
import config
# import dataset
from utils.model_utils import plot_loss, load_model_if_checkpointed, save_model_checkpoint
from models.simpleCNN import SimpleCNN
from models.lettuceNet import LettuceNet
# from models.segmentationNet import SegmentationNet
from torch.utils.tensorboard import SummaryWriter
from utils.datasetSegmentation import DataLoaderSegmentation
import json
import pickle
import pandas as pd
from PIL import Image
import cv2
import segmentation_models_pytorch as smp
import albumentations as albu
from label import Label
import matplotlib.pyplot as plt
from transforms import get_training_augmentation, get_validation_augmentation


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def plot_image(img):
    plt.axis("off")
    plt.imshow(img, origin='upper')
    plt.show()


def plot_seg(input_image, gt_masks, predicted_masks):
    # create a color pallette, selecting a color for each class
    # palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 5 - 1])

    palette = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (0, 255, 255),
    }

    # colors = torch.as_tensor([i for i in range(5)])[:, None] * palette
    # colors = (colors % 255).numpy().astype("uint8")
    image = np.zeros((4, 3), dtype=np.uint8)

    fig = plt.figure(figsize=(50, 50))  # width, height in inches

    for i in range(5):
        sub = fig.add_subplot(1, 5, i + 1)
        sub.imshow(predicted_masks[i, :, :], interpolation='nearest')
        sub = fig.add_subplot(2, 5, i+1)
        sub.imshow(gt_masks[i, :, :], interpolation='nearest')


    # for j in range(4):
    #     image[j] = palette[np.argmax(gt_masks[j])]
    # # plot the semantic segmentation predictions of 5 classes in each color
    # r = Image.fromarray(predicted_masks).resize(input_image.size)
    # r.putpalette(colors)
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(r)
    plt.show()
    print('hello')

# Invert the scaled features to original space
def invert_scaling(data):
    # The model outputs scaled data.. Need to invert their scaling.
    scaler = pickle.load(open(f"{config.SCALER_PATH}{config.SCALERFILE}", 'rb'))
    return scaler.inverse_transform(data)


# Convert the predictions to JSON
def convert_to_json(predictions, image_paths):
    predictions = invert_scaling(predictions)

    data = {}
    data["Measurements"] = {}

    for idx, image_path in enumerate(image_paths):
        image_predictions = predictions[idx, :]
        image_name = image_path.split('/')[-1]
        image_num = image_name.split('.')[0].split('_')[-1]
        prediction_object = {
            "RGBImage": image_name,
            "DebthInformation": f"Debth_{image_num}.png",
            "FreshWeightShoot": image_predictions[0].astype('float'),
            "DryWeightShoot": image_predictions[1].astype('float'),
            "Height": image_predictions[2].astype('float'),
            "Diameter": image_predictions[3].astype('float'),
            "LeafArea": image_predictions[4].astype('float')
        }
        data["Measurements"][f"Image{idx+1}"] = prediction_object

    with open(f"{config.SAVE_PREDICTIONS_DIR}/Images.json", 'w+') as outfile:
        json_dumps_str = json.dumps(data, indent=4)
        print(json_dumps_str, file=outfile)


# Compute the NMSE
def compute_criteria(targets, predictions):

    targets = invert_scaling(targets)
    predictions = invert_scaling(predictions)

    targets_df = pd.DataFrame(targets, columns=config.FEATURES)
    predictions_df = pd.DataFrame(predictions, columns=config.FEATURES)

    error_log = {}
    for column in targets_df.columns:
        # y => ground truths and y_hat => predictions
        error_log[column] = sum([(y - y_hat)**2 for y, y_hat in zip(targets_df[column].values, predictions_df[column].values)])/sum([y**2 for y in targets_df[column].values])

    return error_log


def run_training():
    # model = SegmentationNet()

    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = [cls for cls in Label]
    ACTIVATION = 'softmax2d'  # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cpu'

    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Fix random seed for reproducibility
    np.random.seed(config.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        print('GPU available... using GPU')
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
    else:
        print("GPU not available, using CPU")

    if config.CHECKPOINT_PATH:
        checkpoint_path = os.path.join(os.getcwd(), config.CHECKPOINT_PATH)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
            print("Output directory is created")

    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
    # criterion = torch.nn.BCEWithLogitsLoss()
    # loss_fn = criterion(output, target)

    loss_fn = smp.utils.losses.DiceLoss()

    # TensorBoard
    writer = SummaryWriter(config.TENSORBOARD)

    f = open(config.JSON_FILE)
    meta_data = json.load(f)
    measurements = meta_data["Measurements"]

    img_paths = glob.glob(f"{config.DATA_DIR}/RGB_*")
    img_paths.sort()
    img_paths = img_paths
    mask_paths = glob.glob(f"{config.TARGET_ANNOT_DIR}/Seg_*")
    mask_paths.sort()
    mask_paths = mask_paths

    # --------------------------------------
    # Build Train Dataloaders
    # --------------------------------------

    train_set = DataLoaderSegmentation(img_paths=img_paths, mask_paths=mask_paths, metadata=config.TRAIN_METADATA, center_crop=None, resize=(224, 224), preprocessing=get_preprocessing(preprocessing_fn), augmentation=get_training_augmentation())

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        # collate_fn=collate_fn
    )

    # -----------------------------
    # Build Validation Dataloaders
    # -----------------------------
    test_set = DataLoaderSegmentation(img_paths=img_paths, mask_paths=mask_paths, metadata=config.TEST_METADATA, center_crop=None, resize=(224,224), preprocessing=get_preprocessing(preprocessing_fn), augmentation=get_validation_augmentation())
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        # collate_fn=collate_fn
    )

    print('\nTrain and Validation DataLoader constructed successfully!')

    # Code to use multiple GPUs (if available)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # # --------------------------------------
    # # Load checkpointed model (if  present)
    # # --------------------------------------
    # if config.DEVICE == "cpu":
    #     load_on_cpu = True
    # else:
    #     load_on_cpu = False
    # model, optimizer, checkpointed_loss, checkpoint_flag = load_model_if_checkpointed(model, optimizer, checkpoint_path,
    #                                                                                   load_on_cpu=load_on_cpu)
    # if checkpoint_flag:
    #     print(f"Checkpoint Found! Loading from checkpoint :: LOSS={checkpointed_loss}")
    # else:
    #     print("Checkpoint Not Found! Training from beginning")
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss_fn,
        metrics=metrics,
        optimizer=optimizer,
        device=config.DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss_fn,
        metrics=metrics,
        device=config.DEVICE,
        verbose=True,
    )


    max_score = 0
    train_loss_per_epoch = []
    validation_loss_data = []
    for epoch in range(config.EPOCHS):
        # training
        # train_targets, train_predictions, train_loss = engine_segmentation.train_fn(model, train_loader, optimizer, loss_fn)

        print('\nEpoch: {}'.format(epoch+1))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(test_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, f"{config.CHECKPOINT_PATH}/best_model.pth")
            print('Model saved!')

        if epoch == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

        # # Save model with final train loss (script to save the best weights?)
        # # if checkpointed_loss != 0.0:
        # #     if train_loss < checkpointed_loss:
        # #         save_model_checkpoint(model, optimizer, train_loss, checkpoint_path)
        # #         checkpointed_loss = train_loss
        # #     else:
        # #         pass
        # # else:
        # #     if len(train_loss_per_epoch) > 0:
        # #         if train_loss < min(train_loss_per_epoch):
        # #             save_model_checkpoint(model, optimizer, train_loss, checkpoint_path)
        # #     else:
        # #         save_model_checkpoint(model, optimizer, train_loss, checkpoint_path)
        #
        # error_log_train = compute_criteria(train_targets, train_predictions)
        #
        # # for feature in error_log_train.keys():
        # #     writer.add_scalar(f"Train/{feature}", error_log_train[feature], epoch)
        #
        # NMSE_error = sum([error_log_train[key] for key in error_log_train.keys()])
        # print(f"\nFinished [Epoch: {epoch + 1}/{config.EPOCHS}]",
        #       "\nTraining Loss: {:.3f} |".format(train_loss),
        #       "Train_NMSE : {:.5f} |".format(NMSE_error),)
        #
        # # validation
        # eval_targets, eval_predictions, eval_loss = engine.eval_fn(model, test_loader, loss_fn)
        #
        # error_log_validation = compute_criteria(eval_targets, eval_predictions)
        #
        # # for feature in error_log_validation.keys():
        # #     writer.add_scalar(f"Test/{feature}", error_log_validation[feature], epoch)
        #
        # NMSE_error = sum([error_log_validation[key] for key in error_log_validation.keys()])
        # print(f"\nFinished [Epoch: {epoch + 1}/{config.EPOCHS}]",
        #       "\nTest Loss: {:.3f} |".format(eval_loss),
        #       "Test_NMSE : {:.3f} |".format(NMSE_error),)
        #
        # # mean_loss = np.mean(train_loss_per_epoch)
        # # # Save the mean_loss value for each video instance to the writer
        # # print(f"Avg Training Loss: {np.mean(mean_loss)} for {config.EPOCHS} epochs")
        #
        # # print(f"Epoch {epoch+1} => Training Loss: {train_loss}, Val Loss: {eval_loss}")
        # # print(f"Epoch {epoch} => Training Loss: {train_loss}")
        # train_loss_per_epoch.append(train_loss)
        # validation_loss_data.append(eval_loss)

    # print(train_dataset[0])
    plot_loss(train_loss_per_epoch, validation_loss_data, plot_path=config.PLOT_PATH)
    print("done")


def generate_prediction():
    # model = SegmentationNet()


    ENCODER = 'se_resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = [cls for cls in Label]
    ACTIVATION = 'softmax2d'  # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cpu'

    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


    # Fix random seed for reproducibility
    np.random.seed(config.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        print('GPU available... using GPU')
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
    else:
        print("GPU not available, using CPU")

    if config.CHECKPOINT_PATH:
        checkpoint_path = os.path.join(os.getcwd(), config.CHECKPOINT_PATH)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
            print("Output directory is created")

    model.to(config.DEVICE)

    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.8, patience=5, verbose=True
    # )

    best_model = torch.load(f"{config.CHECKPOINT_PATH}/best_model.pth", map_location=torch.device(config.DEVICE))

    img_paths = glob.glob(f"{config.PREDICTION_DATA_DIR}/RGB_*")
    img_paths.sort()
    img_paths = img_paths[4:8]
    mask_paths = glob.glob(f"{config.TARGET_ANNOT_DIR}/Seg_*")
    mask_paths.sort()
    mask_paths = mask_paths[4:8]
    # --------------------------------------
    # Build Train Dataloaders
    # --------------------------------------
    # test_set = DataLoaderSegmentation(img_paths=img_paths, mask_paths=mask_paths, metadata=config.TEST_METADATA, center_crop=None, resize=(224,224), preprocessing=get_preprocessing(preprocessing_fn))

    prediction_set = DataLoaderSegmentation(img_paths=img_paths, mask_paths=mask_paths, metadata=config.TEST_METADATA, center_crop=None, resize=(224,224), preprocessing=get_preprocessing(preprocessing_fn))

    # prediction_loader = torch.utils.data.DataLoader(
    #     dataset=prediction_set,
    #     batch_size=config.BATCH_SIZE,
    #     num_workers=config.NUM_WORKERS,
    #     shuffle=False,
    #     # collate_fn=collate_fn
    # )

    print('\nPrediction Data loaded!')

    # Code to use multiple GPUs (if available)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    for i in range(4):
        n = np.random.choice(len(prediction_set))

        image_vis = prediction_set[n][0].astype('uint8')
        image, gt_mask = prediction_set[n]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to(config.DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        # input_image, gt_masks, predicted_masks
        plot_seg(
            input_image=image,
            gt_masks=gt_mask,
            predicted_masks=pr_mask
        )

        visualize(
            image=image_vis,
            ground_truth_mask=gt_mask,
            predicted_mask=pr_mask
        )

    # # --------------------------------------
    # # Load checkpointed model (if  present)
    # # --------------------------------------
    # if config.DEVICE == "cpu":
    #     load_on_cpu = True
    # else:
    #     load_on_cpu = False
    # model, optimizer, checkpointed_loss, checkpoint_flag = load_model_if_checkpointed(model, optimizer, checkpoint_path,
    #                                                                                   load_on_cpu=load_on_cpu)
    # if checkpoint_flag:
    #     print(f"Checkpoint Found! Loading from checkpoint :: LOSS={checkpointed_loss}")
    # else:
    #     print("Checkpoint Not Found!")
    #
    # predictions = engine_segmentation.predict_fn(model, prediction_loader)
    # plot_seg(predictions[0], input_image=cv2.imread(img_paths[0]))
    #
    # # Convert predictions to JSON
    # convert_to_json(predictions, img_paths)
    # print("done")


if __name__ == '__main__':
    run_training()
    # generate_prediction()
