IMAGE_FOLDER = "./Data/"  # Folder where all the images will be stored
# ORIGINAL_DATA = "./Data/"

TEST_IMAGES = "./Data/BBDDTest"  # Folder where all test images are stored (user selection)

NUMBER_OF_CLUSTERS = 7  # Number of clusters created in training phase (user selection, PARDINUS uses 7)

TRAINED_MODELS_ROUTE = "./TrainedModels/"  # Folder where trained models will be stored (user selection)

# DATA_NAME = "Animales"  # You have to adjust this variable

# Image features
IMG_WIDTH = 384
IMG_HEIGHT = 256
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3) # RGB color model

# Block size
BLOCK_HEIGHT = 4
BLOCK_WIDTH = 6

# KMEANS_ROUTE = "./TrainedModels/KMeansModel.pkl"
# POST_CLUSTERING_DIRECTORY_NAME = "./Data/BBDD_Clustered_AnimalTest"

RAE_ROUTE = "./TrainedModels/"
SEED = 1491

ERRORS_DIRECTORY = "./ErrorFiles"  # TODO: Borrar esto y aplicar el de abajo
ERROR_FILES_ROUTE = "./ErrorFiles/"

# RF_ROUTE = "./TrainedModels/RandomForestModel.pickle"


