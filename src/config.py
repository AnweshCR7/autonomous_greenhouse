
# Common settings
lr = 0.001
FEATURES = ['FreshWeightShoot', 'DryWeightShoot', 'Height', 'Diameter', 'LeafArea']

## UNCOMMENT IF USING COLAB
# DATA_DIR = "/content/drive/MyDrive/data/FirstTrainingData"
# JSON_FILE = "/content/drive/MyDrive/data/FirstTrainingData/GroundTruth.json"
# TRAIN_METADATA = "/content/drive/MyDrive/data/metadata/TrainGroundTruth.csv"
# TEST_METADATA = "/content/drive/MyDrive/data/metadata/TestGroundTruth.csv"
# SCALER_PATH = "/content/drive/MyDrive/data/metadata/"
# SCALERFILE = "scaler.sav"
# CHECKPOINT_NAME = "running_model_1"
# CHECKPOINT_PATH = "/content/drive/MyDrive/data/GreenHouse/checkpoint"
# PLOT_PATH = "/content/drive/MyDrive/data/GreenHouse/plots"
# NUM_WORKERS = 1
# DEVICE = "cuda"
# BATCH_SIZE = 32
# EPOCHS = 10
# RANDOM_SEED = 42
# PREDICTION_DATA_DIR = "/content/drive/MyDrive/data/GreenHouse/ImagesFor_June7"
# SAVE_PREDICTIONS_DIR = "./"
# TENSORBOARD = "/content/drive/MyDrive/data/GreenHouse/runs"

## COMMENT IF USING COLAB
DATA_DIR = "../data/FirstTrainingData"
JSON_FILE = "../data/FirstTrainingData/GroundTruth.json"
TRAIN_METADATA = "../data/metadata/TrainGroundTruth.csv"
TEST_METADATA = "../data/metadata/TestGroundTruth.csv"
SCALER_PATH = "../data/metadata/"
SCALERFILE = "scaler.sav"
CHECKPOINT_PATH = "../checkpoint"
CHECKPOINT_NAME = "resnet50_pretrained"
PLOT_PATH = "../plots"
BATCH_SIZE = 8
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 75
NUM_WORKERS = 0
DEVICE = "cpu"
EPOCHS = 2
RANDOM_SEED = 42
TENSORBOARD = "./runs"

PREDICTION_DATA_DIR = "../data/ImagesFor_June7"
SAVE_PREDICTIONS_DIR = "./"