IMAGE_FOLDER = "./Data/"  # Folder where all the images will be stored


TEST_IMAGES = "./Data/BBDDTest"  # Folder where all test images are stored (user selection)


TRAINED_MODELS_ROUTE = "./TrainedModels/"  # Folder where trained models will be stored (user selection)


ERROR_FILES_ROUTE = "./ErrorFiles/"  # Folder where the error files will be stored. (user selection)

# Image features
IMG_WIDTH = 384
IMG_HEIGHT = 256
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3) # RGB color model

# Number of blocks to calculate reconstruction error
BLOCK_HEIGHT = 4
BLOCK_WIDTH = 6

NUMBER_OF_CLUSTERS = 7  # Number of clusters created in training phase (user selection, PARDINUS uses 7)

SEED = 1491





