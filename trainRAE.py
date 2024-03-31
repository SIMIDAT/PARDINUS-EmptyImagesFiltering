import config
import utils
from utils import checkGPU
from utils import correntropy

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras




        

def main():

    # Path to equalized empty images (training)
    trainFolder = config.EMPTY_EQUALIZED_TRAINING_IMAGES_PATH

    # Image features
    input_shape = config.INPUT_SHAPE

    # Type of AE
    model = utils.getRAEModel(input_shape)
    

    # Training parameters
    batch_size = config.BATCH_SIZE
    epoch = config.EPOCHS
    verbose = config.VERBOSE
    seed = config.SEED

    numberOfClusters = config.NUMBER_OF_CLUSTERS



    # 1- Check if GPU is available
    checkGPU()

    # 2- Read datasets
    print("Training AE for each cluster")
    print("Parameters")
    print(batch_size, epoch)
    print(trainFolder)

    for clusterIndex in range(numberOfClusters):
        print("\n\n\nChecking...")
        print(trainFolder + str(clusterIndex))

        print("Generating training and validation datasets...")

        trainDataset = ImageDataGenerator(rescale=1./255, data_format='channels_last', validation_split=0.2)

        trainGenerator = trainDataset.flow_from_directory(
            trainFolder + str(clusterIndex),
            target_size = (config.IMG_HEIGHT, config.IMG_WIDTH),
            batch_size=batch_size,
            class_mode='input',
            shuffle=True,
            seed=seed,
            subset='training'
        )

        validationGenerator = trainDataset.flow_from_directory(
            trainFolder + str(clusterIndex),
            target_size = (config.IMG_HEIGHT, config.IMG_WIDTH),
            batch_size=batch_size,
            class_mode='input',
            shuffle=True,
            seed=seed,
            subset='validation'
        )


        # 3- Creation and compilation

        AEname = utils.getRAEName(numberOfClusters, clusterIndex)
        

        print("\n----------------------------")
        print("Processing AE -> ", AEname)

        
        model.compile(
            optimizer='adam',
            loss=correntropy,
            metrics=['mse']
        )

        # 4- Fit

        history = model.fit(
            trainGenerator,
            validation_data = validationGenerator,
            epochs = epoch,
            verbose=verbose
        )

        print(history.history.keys())

        
        # 5- Save models

        # RAE save
        model.save_weights(AEname + ".h5")



if __name__ == '__main__':
    main()