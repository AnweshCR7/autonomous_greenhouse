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
from utils.model_utils import plot_loss
from models.simpleCNN import SimpleCNN
from models.lettuceNet import LettuceNet
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import DataLoaderLettuceNet
import json


def run_training():
    model = LettuceNet()

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
    loss_fn = nn.L1Loss()

    # TensorBoard
    writer = SummaryWriter()

    # A simple transform which we will apply to the MNIST images
    # simple_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    # train_set = datasets.MNIST(root=config.DATA_DIR, train=True, transform=simple_transform, download=True)
    # test_set = datasets.MNIST(root=config.DATA_DIR, train=False, transform=simple_transform, download=True)

    # train_dataset = dataset.ClassificationDataset(image_paths=train_imgs, targets=train_targets,
    #                                               resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))

    f = open(config.JSON_FILE)
    meta_data = json.load(f)
    measurements = meta_data["Measurements"]

    img_paths = glob.glob(f"{config.DATA_DIR}/RGB_*")
    # Ad hoc at this point
    train_img_paths = img_paths[:180]
    test_img_paths = img_paths[180:]

    # --------------------------------------
    # Build Train Dataloaders
    # --------------------------------------

    train_set = DataLoaderLettuceNet(img_paths=train_img_paths, meta_data=measurements, center_crop=700, resize=(224,224))

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
    test_set = DataLoaderLettuceNet(img_paths=test_img_paths, meta_data=measurements, center_crop=700, resize=(224,224))
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
        # collate_fn=collate_fn
    )

    print('\nTrain and Validation DataLoader constructed successfully!')

    train_loss_data = []
    validation_loss_data = []
    for epoch in range(config.EPOCHS):
        # training
        train_loss = engine.train_fn(model, train_loader, optimizer, loss_fn)

        # validation
        eval_preds, eval_loss = engine.eval_fn(model, test_loader, loss_fn)

        print(f"Epoch {epoch} => Training Loss: {train_loss}, Val Loss: {eval_loss}")
        # print(f"Epoch {epoch} => Training Loss: {train_loss}")
        train_loss_data.append(train_loss)
        validation_loss_data.append(eval_loss)

    # print(train_dataset[0])
    plot_loss(train_loss_data, validation_loss_data, plot_path=config.PLOT_PATH)
    print("done")


if __name__ == '__main__':
    run_training()
