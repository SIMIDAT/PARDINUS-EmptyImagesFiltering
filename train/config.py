IMAGE_FOLDER = "./Data/"  # Folder where all the images will be stored

EMPTY_DATA = IMAGE_FOLDER + "BBDDTrain/Empty"  # Route to empty training data. It must be inside IMAGE_FOLDER (user selection)
ANIMAL_DATA = IMAGE_FOLDER + "BBDDTrain/Animal"  # Route to animal data. It must be inside IMAGE_FOLDER (user selection)


TRAINED_MODELS_ROUTE = "./TrainedModels/"  # Folder where trained models will be stored (user selection)

ERROR_FILES_ROUTE = "./ErrorFiles/"




# Image features
IMG_WIDTH = 384
IMG_HEIGHT = 256
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3) # RGB color model

NUMBER_OF_CLUSTERS = 7  # Number of clusters to create

# Block size
BLOCK_HEIGHT = 4
BLOCK_WIDTH = 6

# Train RAE
BATCH_SIZE = 16
EPOCHS = 70
VERBOSE = 1
SEED = 1491

ANIMAL_PROPORTION = 24  # Proportion of animal images vs empty images (i.e 24 means that 24% of all images are non-empty)

# Train RF
NUMBER_OF_ESTIMATORS = 200