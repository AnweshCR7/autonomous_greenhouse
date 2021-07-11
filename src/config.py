
# Common settings
FEATURES = ['FreshWeightShoot', 'DryWeightShoot', 'LeafArea']
ADD_FEATURES = ['Diameter', 'Height', 'Area', 'Volume']

# ------------------------------------------------------------
# ATTENTION HARRY: You probably need to change these variables
# ------------------------------------------------------------

# CSV with the height/diameter predictions
HARRY_FEATURES_FOR_PREDICTION = "../data/features/X_eval.csv"
# Data directory where new prediction images are stored.
PREDICTION_DATA_DIR = "../data/FirstTrainingData"
# Data Directory where segmentations are stored
SEG_DIR = "../data/final_data/"
# File saved as ./predictions/Images.json
SAVE_PREDICTIONS_DIR = "./predictions/"

# ------------------------------------------------------------
# AND LETS GO!!
# ------------------------------------------------------------

# ---- Model Variables ----
lr = 0.001
BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = "cpu"
EPOCHS = 2
RANDOM_SEED = 34
TENSORBOARD = "./runs"
# Keep this False (use in case you have the GT for inference)
ONLY_EVAL = False

SCALER_PATH = "../data/metadata/"
SCALERFILE = "scaler.sav"
SCALERFILE_X = "ft_scaler_x.sav"
SCALERFILE_Y = "ft_scaler_y.sav"

# Contains the Latest Ground Truth (all images)
MASTER_METADATA = "../data/master_data.csv"
# Latest features from Harry's model
MASTER_METADATA_ADD_FT = "../data/final_data/Feature_all.csv"
# Data directory where all images are stored.
DATA_DIR = "../data/FirstTrainingData"

# CSVs needed for the evaluation (No need to change for prediction)
TEST_ADD_FEATURES = "../data/features/X_eval.csv"
# In case we have the Ground Truths
TEST_ADD_FEATURES_Y = "../data/features/y_eval.csv"

# Checkpoint
CHECKPOINT_PATH = "../checkpoint"
CHECKPOINT_NAME = "direct_reg_effnet_150"


# # # UNCOMMENT IF USING COLAB
# # New vesion files"
# MASTER_METADATA = "/content/drive/MyDrive/data/GreenHouse/metadata/master_data.csv"
# MASTER_METADATA_ADD_FT = "/content/drive/MyDrive/data/GreenHouse/final_data/Feature_all.csv"
# # ""
# DATA_DIR = "/content/drive/MyDrive/data/GreenHouse/FirstTrainingData"
# JSON_FILE = "/content/drive/MyDrive/data/GreenHouse/FirstTrainingData/GroundTruth.json"
# TRAIN_METADATA = "/content/drive/MyDrive/data/GreenHouse/metadata/TrainGroundTruth_seg.csv"
# TEST_METADATA = "/content/drive/MyDrive/data/GreenHouse/metadata/TestGroundTruth_seg.csv"
# TRAIN_ADD_FEATURES = "/content/drive/MyDrive/data/GreenHouse/metadata/X_train.csv"
# TRAIN_ADD_FEATURES_Y = "/content/drive/MyDrive/data/GreenHouse/metadata/y_train.csv"
# TEST_ADD_FEATURES = "/content/drive/MyDrive/data/GreenHouse/metadata/X_eval.csv"
# TEST_ADD_FEATURES_Y = "/content/drive/MyDrive/data/GreenHouse/metadata/y_eval.csv"
# SEG_DIR = "/content/drive/MyDrive/data/GreenHouse/final_data"
# SCALER_PATH = "/content/drive/MyDrive/data/GreenHouse/metadata/"
# SCALERFILE = "scaler.sav"
# SCALERFILE_X = "ft_scaler_x.sav"
# SCALERFILE_Y = "ft_scaler_y.sav"
# CHECKPOINT_NAME = "final_img_reg"
# CHECKPOINT_PATH = "/content/drive/MyDrive/data/GreenHouse/checkpoint"
# PLOT_PATH = "/content/drive/MyDrive/data/GreenHouse/plots"
# NUM_WORKERS = 1
# DEVICE = "cuda"
# BATCH_SIZE = 32
# EPOCHS = 30
# RANDOM_SEED = 23
# PREDICTION_DATA_DIR = "/content/drive/MyDrive/data/GreenHouse/ImagesFor_July5"
# SAVE_PREDICTIONS_DIR = "/content/drive/MyDrive/data/GreenHouse/predictions"
# TENSORBOARD = "/content/drive/MyDrive/data/GreenHouse/runs"
# PRED_SEG_DIR = "/content/drive/MyDrive/data/GreenHouse/segmentation_5_july"
# PRED_ADD_FEATURES = "/content/drive/MyDrive/data/GreenHouse/metadata/Feature_eval_ImagesFor_July5.csv"
# ONLY_EVAL = False

# # COMMENT IF USING COLAB