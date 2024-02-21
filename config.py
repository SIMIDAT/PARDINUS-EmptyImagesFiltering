# Image features
IMG_WIDTH = 384
IMG_HEIGHT = 256
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3) # RGB color model

# Block size
BLOCK_HEIGHT = 4
BLOCK_WIDTH = 6

# Set if training or testing models
'''
Training or testing

- "Train" = Training
- "Test" = Testing
'''
TRAINTEST = "Train"

# Clustering
NUMBER_OF_CLUSTERS = 7

# Images folders
EMPTY_EQUALIZED_TRAINING_IMAGES_PATH = "./Database/Equalized_Clustered/Train/Equalized_Clustered_Empty_Images"
ANIMAL_EQUALIZED_TRAINING_IMAGES_PATH = "./Database/Equalized_Clustered/Train/Equalized_Clustered_Animal_Images"

EQUALIZED_TEST_IMAGES_PATH = "./Database/Equalized_Clustered/Test/Equalized_Clustered_Images"

# Train RAE
BATCH_SIZE = 16
EPOCHS = 70
VERBOSE = 1
SEED = 1491

# Balance error file
BALANCE = True
ANIMAL_PROPORTION = 22  # (training animal images / training empty images) * 100