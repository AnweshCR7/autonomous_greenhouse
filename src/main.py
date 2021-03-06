import os
import glob
import torch
import numpy as np
import torch.nn as nn
import engine
import config
# import dataset
from utils.model_utils import plot_loss, load_model_if_checkpointed, save_model_checkpoint
from models.lettuceNetPlus import LettuceNetPlus
from models.lettuceNetEff import LettuceNetEff
from models.effNetDirect import LettuceNetEffDirect
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

    if not os.path.exists(config.SAVE_PREDICTIONS_DIR):
        os.makedirs(config.SAVE_PREDICTIONS_DIR)

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
    # model = LettuceNetPlus()
    model = LettuceNetEffDirect()
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

    master_metadata_csv = config.MASTER_METADATA
    master_metadata_add_features = config.MASTER_METADATA_ADD_FT

    # This csv is just there to read Harry's eval_set and eventually filter the training data from the master dataset.
    test_add_features_csv = config.TEST_ADD_FEATURES

    test_img_paths = []
    all_img_paths = []
    test_img_numbers = []

    test_df = pd.read_csv(test_add_features_csv)

    master_df = pd.read_csv(master_metadata_csv)

    # Get the names of all the test images.
    # But we will only take the image names from this dataframe.
    # The features will be taken from the master dataset.
    for image_name in test_df["Unnamed: 0"].values:
        image_index = int(re.findall("\d+", image_name)[0])
        test_img_paths.append(f"{config.DATA_DIR}/RGB_{image_index}.png")
        test_img_numbers.append(image_index)

    # New code with Harry's GT
    for image_name in master_df["ImageName"].values:
        image_index = int(re.findall("\d+", image_name)[0])
        all_img_paths.append(f"{config.DATA_DIR}/RGB_{image_index}.png")

    # Now filter the training set from the master set
    train_img_paths = [path for path in all_img_paths if int(re.findall("\d+", path)[0]) not in test_img_numbers]

    # --------------------------------------
    # Build Train Dataloaders
    # --------------------------------------

    train_set = DataLoaderLettuceNet(img_paths=train_img_paths, metadata=master_metadata_csv, center_crop=(960, 810), resize=(224, 224), add_features=master_metadata_add_features, augmentations="train")

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
    test_set = DataLoaderLettuceNet(img_paths=test_img_paths, metadata=master_metadata_csv, center_crop=(960, 810), resize=(224,224), add_features=master_metadata_add_features, augmentations="validation")
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
        if not config.ONLY_EVAL:
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
        # else:
        # validation
        eval_targets, eval_predictions, eval_loss = engine.eval_fn(model, test_loader, loss_fn)

        error_log_validation = compute_criteria(eval_targets, eval_predictions, save=True, img_meta=test_df["Unnamed: 0"].values)

        for feature in error_log_validation.keys():
            writer.add_scalar(f"Test/{feature}", error_log_validation[feature], epoch)
            print(f"Test/{feature}: {error_log_validation[feature]}")

        NMSE_error = sum([error_log_validation[key] for key in error_log_validation.keys()])
        print(f"\nFinished [Epoch: {epoch + 1}/{config.EPOCHS}]",
              "\nTest Loss: {:.3f} |".format(eval_loss),
              "Test_NMSE : {:.3f} |".format(NMSE_error),)

        validation_loss_data.append(eval_loss)
        scheduler.step(NMSE_error)

    print("done")


# Still need to fix the prediict function to work with the master dataset...
def generate_prediction():
    model = LettuceNetEffDirect()

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

    # The Y to be used when we have the ground truth
    test_metadata_csv = config.TEST_ADD_FEATURES_Y
    # test_y = pd.read_csv(config.TEST_ADD_FEATURES_Y)
    # The X in addition to the Image
    test_add_features_csv = config.HARRY_FEATURES_FOR_PREDICTION
    test_df = pd.read_csv(test_add_features_csv)


    for image_name in test_df["Unnamed: 0"].values:
        image_index = int(re.findall("\d+", image_name)[0])
        prediction_img_paths.append(f"{config.PREDICTION_DATA_DIR}/RGB_{image_index}.png")

    # --------------------------------------
    # Build Prediction Dataloaders
    # --------------------------------------

    prediction_set = DataLoaderLettuceNet(img_paths=prediction_img_paths, metadata=test_metadata_csv, center_crop=(960, 810), resize=(224, 224), predict=True,
                                     add_features=test_add_features_csv, augmentations="val")

    prediction_loader = torch.utils.data.DataLoader(
        dataset=prediction_set,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
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
    print("done")


if __name__ == '__main__':
    # run_training()
    generate_prediction()
