IMAGE_FOLDER = "./Data/"  # Folder where all the images will be stored

EMPTY_DATA = IMAGE_FOLDER + "BBDDTrain/Vacio"  # Route to empty training data. It must be inside IMAGE_FOLDER (user selection)
ANIMAL_DATA = IMAGE_FOLDER + "BBDDTrain/Animales"  # Route to animal data. It must be inside IMAGE_FOLDER (user selection)

NUMBER_OF_CLUSTERS = 7  # Number of clusters to create (user selection)
TRAINED_MODELS_ROUTE = "./TrainedModels/"  # Folder where trained models will be stored (user selection)

ERROR_FILES_ROUTE = "./ErrorFiles/"


TRAINING_DATA = "./Data/BBDDTrain"
TEST_DATA = "./Data/BBDDTest"

ANIMAL_TEST_DATA = "./Data/BBDDTest/Animales"
EMPTY_TEST_DATA = "./Data/BBDDTest/Vacio"








# KMEANS_ROUTE = "./TrainedModels/KMeansModel.pkl"  #
RAE_ROUTE = "./TrainedModels/"

# POST_CLUSTERING_DIRECTORY_NAME = "./Data/BBDD_Clustered_EmptyTrain"  # Vacías de entrenamiento
# POST_CLUSTERING_DIRECTORY_NAME_ANIMALTRAIN = "./Data/BBDD_Clustered_AnimalTrain"


# POST_CLUSTERING_DIRECTORY_NAME_EMPTYTEST = "./Data/BBDD_Clustered_EmptyTest"
# POST_CLUSTERING_DIRECTORY_NAME_ANIMALTEST = "./Data/BBDD_Clustered_AnimalTest"


EMPTY_TRAIN_ERROR_FILE = "./ErrorFiles/Train_Errors_7_Vacio.txt"
ANIMAL_TRAIN_ERROR_FILE = "./ErrorFiles/Train_Errors_7_Animales.txt"

IMG_WIDTH = 384
IMG_HEIGHT = 256
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3) # RGB color model

# Block size
BLOCK_HEIGHT = 4
BLOCK_WIDTH = 6

# Train RAE
BATCH_SIZE = 16
EPOCHS = 70
VERBOSE = 1
SEED = 1491