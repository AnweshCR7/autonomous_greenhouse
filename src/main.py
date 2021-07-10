import os
import glob
import torch
import numpy as np
from pprint import pprint
from torchvision import datasets, transforms
import torch.nn as nn
from sklearn import model_selection
from sklearn import metrics
import engine
import config
# import dataset
from utils.model_utils import plot_loss, load_model_if_checkpointed, save_model_checkpoint
from models.simpleCNN import SimpleCNN
from models.lettuceNet import LettuceNet
from models.lettuceNetPlus import LettuceNetPlus
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import DataLoaderLettuceNet
import json
import pickle
import pandas as pd
import re


# Invert the scaled features to original space
def invert_scaling(data):
    # The model outputs scaled data.. Need to invert their scaling.
    scaler = pickle.load(open(f"{config.SCALER_PATH}{config.SCALERFILE_Y}", 'rb'))
    return scaler.inverse_transform(data)


# Convert the predictions to JSON
def convert_to_json(predictions, image_paths, test_df):
    predictions = invert_scaling(predictions)

    data = {}
    data["Measurements"] = {}

    for idx, image_path in enumerate(image_paths):
        image_predictions = predictions[idx, :]
        image_name = image_path.split('/')[-1]
        image_num = image_name.split('.')[0].split('_')[-1]
        rem_features = test_df[test_df['Unnamed: 0'] == f"Image{image_num}"]

        prediction_object = {
            "RGBImage": image_name,
            "DebthInformation": f"Debth_{image_num}.png",
            "FreshWeightShoot": image_predictions[0].astype('float'),
            "DryWeightShoot": image_predictions[1].astype('float'),
            # from test_df
            "Height": rem_features["Height"].values[0].astype('float'),
            "Diameter": rem_features["Diameter"].values[0].astype('float'),
            "LeafArea": image_predictions[2].astype('float')
        }
        data["Measurements"][f"Image{idx+1}"] = prediction_object

    with open(f"{config.SAVE_PREDICTIONS_DIR}/Images.json", 'w+') as outfile:
        json_dumps_str = json.dumps(data, indent=4)
        print(json_dumps_str, file=outfile)


# Compute the NMSE
def compute_criteria(targets, predictions, save=False, img_meta=None):

    targets = invert_scaling(targets)
    predictions = invert_scaling(predictions)

    targets_df = pd.DataFrame(targets, columns=config.FEATURES)
    predictions_df = pd.DataFrame(predictions, columns=config.FEATURES)

    error_log = {}
    for column in targets_df.columns:
        # y => ground truths and y_hat => predictions
        error_log[column] = sum([(y - y_hat)**2 for y, y_hat in zip(targets_df[column].values, predictions_df[column].values)])/sum([y**2 for y in targets_df[column].values])

    if save:
        targets_df["ImageName"] = pd.DataFrame(img_meta)
        targets_df.to_csv(f"{config.SAVE_PREDICTIONS_DIR}/targets_df.csv", index=False)
        predictions_df["ImageName"] = pd.DataFrame(img_meta)
        predictions_df.to_csv(f"{config.SAVE_PREDICTIONS_DIR}/preds_df.csv", index=False)

    return error_log


def run_training():
    model = LettuceNetPlus()

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
    loss_fn = nn.MSELoss()

    # TensorBoard
    writer = SummaryWriter(config.TENSORBOARD)

    f = open(config.JSON_FILE)
    meta_data = json.load(f)
    measurements = meta_data["Measurements"]

    train_metadata_csv = config.TRAIN_ADD_FEATURES_Y
    test_metadata_csv = config.TEST_ADD_FEATURES_Y
    train_add_features_csv = config.TRAIN_ADD_FEATURES
    test_add_features_csv = config.TEST_ADD_FEATURES

    # img_paths = glob.glob(f"{config.DATA_DIR}/RGB_*")
    train_img_paths = []
    test_img_paths = []
    # Ad hoc at this point
    # train_df = pd.read_csv(train_metadata_csv)
    # test_df = pd.read_csv(test_metadata_csv)
    train_df = pd.read_csv(train_add_features_csv)
    test_df = pd.read_csv(test_add_features_csv)

    # New code with Harry's GT
    for image_name in train_df["Unnamed: 0"].values:
        image_index = int(re.findall("\d+", image_name)[0])
        train_img_paths.append(f"{config.DATA_DIR}/RGB_{image_index}.png")

    # train_img_paths = train_img_paths[:16]

    for image_name in test_df["Unnamed: 0"].values:
        image_index = int(re.findall("\d+", image_name)[0])
        test_img_paths.append(f"{config.DATA_DIR}/RGB_{image_index}.png")
    # test_img_paths = ["../data/FirstTrainingData/RGB_309.png"]



    # Old code with full GT
    # for image_name in train_df["ImageName"].values:
    #     image_index = int(re.findall("\d+", image_name)[0])
    #     train_img_paths.append(f"{config.DATA_DIR}/RGB_{image_index}.png")
    #     # train_img_paths.append(f"{config.DATA_DIR}/hf_RGB_{image_index}.png")
    #     # train_img_paths.append(f"{config.DATA_DIR}/vf_RGB_{image_index}.png")
    #     # train_img_paths.append(f"{config.DATA_DIR}/vfhf_RGB_{image_index}.png")
    #
    # # train_img_paths = train_img_paths[:16]
    #
    # for image_name in test_df["ImageName"].values:
    #     image_index = int(re.findall("\d+", image_name)[0])
    #     test_img_paths.append(f"{config.DATA_DIR}/RGB_{image_index}.png")
    #     # test_img_paths.append(f"{config.DATA_DIR}/vf_RGB_{image_index}.png")
    #     # test_img_paths.append(f"{config.DATA_DIR}/hf_RGB_{image_index}.png")
    #     # test_img_paths.append(f"{config.DATA_DIR}/vfhf_RGB_{image_index}.png")

    # segmentation_paths = glob.glob(f"{config.SEG_DIR}/*.png")

    # --------------------------------------
    # Build Train Dataloaders
    # --------------------------------------

    train_set = DataLoaderLettuceNet(img_paths=train_img_paths, metadata=train_metadata_csv, center_crop=(960, 810), resize=(224, 224), add_features=train_add_features_csv, augmentations="train")

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
    test_set = DataLoaderLettuceNet(img_paths=test_img_paths, metadata=test_metadata_csv, center_crop=(960, 810), resize=(224,224), add_features=test_add_features_csv, augmentations="validation")
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

    # --------------------------------------
    # Load checkpointed model (if  present)
    # --------------------------------------
    if config.DEVICE == "cpu":
        load_on_cpu = True
    else:
        load_on_cpu = False
    model, optimizer, checkpointed_loss, checkpoint_flag = load_model_if_checkpointed(model, optimizer, checkpoint_path,
                                                                                      load_on_cpu=load_on_cpu)
    if checkpoint_flag:
        print(f"Checkpoint Found! Loading from checkpoint :: LOSS={checkpointed_loss}")
    else:
        print("Checkpoint Not Found! Training from beginning")


    train_loss_per_epoch = []
    validation_loss_data = []
    for epoch in range(config.EPOCHS):
        # training
        train_targets, train_predictions, train_loss = engine.train_fn(model, train_loader, optimizer, loss_fn)

        # Save model with final train loss (script to save the best weights?)
        if checkpointed_loss != 0.0:
            if train_loss < checkpointed_loss:
                save_model_checkpoint(model, optimizer, train_loss, checkpoint_path)
                checkpointed_loss = train_loss
            else:
                pass
        else:
            if len(train_loss_per_epoch) > 0:
                if train_loss < min(train_loss_per_epoch):
                    save_model_checkpoint(model, optimizer, train_loss, checkpoint_path)
            else:
                save_model_checkpoint(model, optimizer, train_loss, checkpoint_path)

        error_log_train = compute_criteria(train_targets, train_predictions)

        for feature in error_log_train.keys():
            writer.add_scalar(f"Train/{feature}", error_log_train[feature], epoch)
            print(f"Train/{feature}: {error_log_train[feature]}")

        NMSE_error = sum([error_log_train[key] for key in error_log_train.keys()])
        print(f"\nFinished [Epoch: {epoch + 1}/{config.EPOCHS}]",
              "\nTraining Loss: {:.3f} |".format(train_loss),
              "Train_NMSE : {:.5f} |".format(NMSE_error),)

        # validation
        eval_targets, eval_predictions, eval_loss = engine.eval_fn(model, test_loader, loss_fn)

        error_log_validation = compute_criteria(eval_targets, eval_predictions)

        for feature in error_log_validation.keys():
            writer.add_scalar(f"Test/{feature}", error_log_validation[feature], epoch)
            print(f"Test/{feature}: {error_log_validation[feature]}")

        NMSE_error = sum([error_log_validation[key] for key in error_log_validation.keys()])
        print(f"\nFinished [Epoch: {epoch + 1}/{config.EPOCHS}]",
              "\nTest Loss: {:.3f} |".format(eval_loss),
              "Test_NMSE : {:.3f} |".format(NMSE_error),)

        # mean_loss = np.mean(train_loss_per_epoch)
        # # Save the mean_loss value for each video instance to the writer
        # print(f"Avg Training Loss: {np.mean(mean_loss)} for {config.EPOCHS} epochs")

        # print(f"Epoch {epoch+1} => Training Loss: {train_loss}, Val Loss: {eval_loss}")
        # print(f"Epoch {epoch} => Training Loss: {train_loss}")
        # train_loss_per_epoch.append(train_loss)
        validation_loss_data.append(eval_loss)

    # print(train_dataset[0])
    plot_loss(train_loss_per_epoch, validation_loss_data, plot_path=config.PLOT_PATH)
    print("done")


def generate_prediction():
    model = LettuceNetPlus()

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
    prediction_img_paths = []

    # The Y
    test_metadata_csv = config.TEST_ADD_FEATURES_Y
    # test_y = pd.read_csv(config.TEST_ADD_FEATURES_Y)
    # The X in addition to the Image
    test_add_features_csv = config.TEST_ADD_FEATURES
    test_df = pd.read_csv(test_add_features_csv)


    for image_name in test_df["Unnamed: 0"].values:
        image_index = int(re.findall("\d+", image_name)[0])
        prediction_img_paths.append(f"{config.PREDICTION_DATA_DIR}/RGB_{image_index}.png")


    # prediction_img_paths = glob.glob(f"{config.PREDICTION_DATA_DIR}/RGB_*")

    # --------------------------------------
    # Build Train Dataloaders
    # --------------------------------------

    # train_set = DataLoaderLettuceNet(img_paths=train_img_paths, metadata=train_metadata_csv, center_crop=(960, 810), resize=(224, 224), add_features=train_add_features_csv, augmentations="train")

    prediction_set = DataLoaderLettuceNet(img_paths=prediction_img_paths, metadata=test_metadata_csv, center_crop=(960, 810), resize=(224, 224), predict=True,
                                     add_features=test_add_features_csv, augmentations="val")

    prediction_loader = torch.utils.data.DataLoader(
        dataset=prediction_set,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        # collate_fn=collate_fn
    )

    print('\nPrediction Data loaded!')

    # Code to use multiple GPUs (if available)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # --------------------------------------
    # Load checkpointed model (if  present)
    # --------------------------------------
    if config.DEVICE == "cpu":
        load_on_cpu = True
    else:
        load_on_cpu = False

    model, optimizer, checkpointed_loss, checkpoint_flag = load_model_if_checkpointed(model, optimizer, checkpoint_path,
                                                                                      load_on_cpu=load_on_cpu)
    if checkpoint_flag:
        print(f"Checkpoint Found! Loading from checkpoint :: LOSS={checkpointed_loss}")
    else:
        print("Checkpoint Not Found!")

    predictions = engine.predict_fn(model, prediction_loader)
    # error_log_validation = compute_criteria(targets, predictions, save=True, img_meta=test_y["Unnamed: 0"].values)

    # Convert predictions to JSON
    convert_to_json(predictions, prediction_img_paths, test_df)
    # for feature in error_log_validation.keys():
    #     print(f"Test/{feature}: {error_log_validation[feature]}")

    # NMSE_error = sum([error_log_validation[key] for key in error_log_validation.keys()])
    # print(f"\nFinished [Test Epoch]",
    #       "Test_NMSE : {:.3f} |".format(NMSE_error), )
    print("done")


if __name__ == '__main__':
    # run_training()
    generate_prediction()
