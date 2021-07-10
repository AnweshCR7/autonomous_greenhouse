
# Common settings
lr = 0.001
FEATURES = ['FreshWeightShoot', 'DryWeightShoot', 'LeafArea']
ADD_FEATURES = ['Diameter', 'Height', 'Area', 'Volume']

# # UNCOMMENT IF USING COLAB
# DATA_DIR = "/content/drive/MyDrive/data/GreenHouse/FirstTrainingData"
# JSON_FILE = "/content/drive/MyDrive/data/GreenHouse/FirstTrainingData/GroundTruth.json"
# TRAIN_METADATA = "/content/drive/MyDrive/data/GreenHouse/metadata/TrainGroundTruth_seg.csv"
# TEST_METADATA = "/content/drive/MyDrive/data/GreenHouse/metadata/TestGroundTruth_seg.csv"
# TRAIN_ADD_FEATURES = "/content/drive/MyDrive/data/GreenHouse/metadata/X_train.csv"
# TRAIN_ADD_FEATURES_Y = "/content/drive/MyDrive/data/GreenHouse/metadata/y_train.csv"
# TEST_ADD_FEATURES = "/content/drive/MyDrive/data/GreenHouse/metadata/X_eval.csv"
# TEST_ADD_FEATURES_Y = "/content/drive/MyDrive/data/GreenHouse/metadata/y_eval.csv"
# SEG_DIR = "/content/drive/MyDrive/data/GreenHouse/segmentation"
# SCALER_PATH = "/content/drive/MyDrive/data/GreenHouse/metadata/"
# SCALERFILE = "scaler.sav"
# SCALERFILE_X = "ft_scaler_x.sav"
# SCALERFILE_Y = "ft_scaler_y.sav"
# CHECKPOINT_NAME = "check10July"
# CHECKPOINT_PATH = "/content/drive/MyDrive/data/GreenHouse/checkpoint"
# PLOT_PATH = "/content/drive/MyDrive/data/GreenHouse/plots"
# NUM_WORKERS = 1
# DEVICE = "cpu"
# BATCH_SIZE = 32
# EPOCHS = 30
# RANDOM_SEED = 23
# PREDICTION_DATA_DIR = "/content/drive/MyDrive/data/GreenHouse/ImagesFor_July5"
# SAVE_PREDICTIONS_DIR = "/content/drive/MyDrive/data/GreenHouse/predictions"
# TENSORBOARD = "/content/drive/MyDrive/data/GreenHouse/runs"
# PRED_SEG_DIR = "/content/drive/MyDrive/data/GreenHouse/segmentation_5_july"
# PRED_ADD_FEATURES = "/content/drive/MyDrive/data/GreenHouse/metadata/Feature_eval_ImagesFor_July5.csv"

# COMMENT IF USING COLAB
DATA_DIR = "../data/FirstTrainingData"
JSON_FILE = "./GroundTruth_SendJuly6.json"
# New vesion files"
MASTER_METADATA = "../data/master_data.csv"
MASTER_METADATA_ADD_FT = "../data/final_data/Feature_all.csv"
# eod
TRAIN_METADATA = "../data/metadata/TrainGroundTruth.csv"
TEST_METADATA = "../data/metadata/TestGroundTruth.csv"
TRAIN_ADD_FEATURES = "../data/features/X_train.csv"
TRAIN_ADD_FEATURES_Y = "../data/features/y_train.csv"
TEST_ADD_FEATURES = "../data/features/X_eval.csv"
TEST_ADD_FEATURES_Y = "../data/features/y_eval.csv"
SCALER_PATH = "../data/metadata/"
SCALERFILE = "scaler.sav"
SCALERFILE_X = "ft_scaler_x.sav"
SCALERFILE_Y = "ft_scaler_y.sav"
CHECKPOINT_PATH = "../checkpoint"
CHECKPOINT_NAME = "adassfsd"
PLOT_PATH = "../plots"
BATCH_SIZE = 32
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 75
NUM_WORKERS = 1
DEVICE = "cpu"
EPOCHS = 2
RANDOM_SEED = 34
TENSORBOARD = "./runs"

PREDICTION_DATA_DIR = "../data/ImagesFor_June7"
SAVE_PREDICTIONS_DIR = "./"
SEG_DIR = "../data/final_data/"