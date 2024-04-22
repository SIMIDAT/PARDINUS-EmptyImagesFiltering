ORIGINAL_DATA = "./Data/"
DATA_NAME = "Vacio"  # You have to adjust this variable

# Image features
IMG_WIDTH = 384
IMG_HEIGHT = 256
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3) # RGB color model

# Block size
BLOCK_HEIGHT = 4
BLOCK_WIDTH = 6

KMEANS_ROUTE = "./TrainedModels/KMeansModel.pkl"
POST_CLUSTERING_DIRECTORY_NAME = "./Data/EqualizedClustered"

NUMBER_OF_CLUSTERS = 7

RAE_ROUTE = "./TrainedModels/"
SEED = 1491

ERRORS_DIRECTORY = "./ErrorFiles"

RF_ROUTE = "./TrainedModels/RandomForestModel.pickle"


