# DATA_DIR = "/content/drive/MyDrive/data/FirstTrainingData"
# JSON_FILE = "/content/drive/MyDrive/data/FirstTrainingData/GroundTruth.json"
# CHECKPOINT_PATH = "/content/drive/MyDrive/data/GreenHouse/checkpoint"
# PLOT_PATH = "/content/drive/MyDrive/data/GreenHouse/plots"
# NUM_WORKERS = 1
# DEVICE = "cuda"
# BATCH_SIZE = 32
# EPOCHS = 10
# RANDOM_SEED = 42

DATA_DIR = "../data/FirstTrainingData"
JSON_FILE = "../data/FirstTrainingData/GroundTruth.json"
CHECKPOINT_PATH = "../checkpoint"
CHECKPOINT_NAME = "running_model_1"
PLOT_PATH = "../plots"
BATCH_SIZE = 8
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 75
NUM_WORKERS = 0
DEVICE = "cpu"
EPOCHS = 2
RANDOM_SEED = 42

lr = 0.001
FEATURES = ['FreshWeightShoot', 'DryWeightShoot', 'Height', 'Diameter', 'LeafArea']