# Weakly supervised discarding of photo-trapping empty images based on autoencoders

This repository is the official implementation of the paper “Weakly supervised discarding of photo-trapping empty images based on autoencoders”. PARDINUS, built on the foundation of weakly-supervised learning, is the proposed tool to automatically detect empty or blank images.


## Install

Install requirements.txt using the following python code in a clean anaconda environment (Python 3.9 is reccomended):

```
pip install -r requirements.txt

```


## Code organization and usage
The code is organized into two main folders: _Train_ and _Test_. _Test_ folder contains the scripts to test or evaluate pretrained models on your own images. _Train_ folder contains the scripts for training your own models.


### Preprocessing
The images must be 256 pixels height and 384 pixels width, RGB format. If your images needs to be resized, you can use the script resizeImages.py


### Test or inference

There are three scripts that you should execute to test trained models on your own images: _clustering.py, autoencoders.py_ and _randomForest.py_, __in this order__. 

```
python clustering.py
python autoencoders.py
python randomForest.py
```


The file _config.py_ contains variables that needs to be set to make the scripts work properly. When indicating the route to a specific folder, you have to create the folder itself, with the name you set. For example, if you set IMAGE_FOLDER = "./MY_IMAGES/", you would have to create a folder in the root named "./MY_IMAGES/", where the images should be stored.

- IMAGE_FOLDER: set where the clustered and equalized images for the RAEs will be stored. Default: "./Data/"
  
- TEST_IMAGES: set where the original images, your own images, are stored. Default: "./Data/BBDDTest"
  
- TRAINED_MODELS_ROUTE: set where the trained models are stored. There should be one k-means clustering model, one random forest model and one RAE model for each cluster of images. Default: "./TrainedModels/"
  
- ERROR_FILES_ROUTE: set where the trained models are stored. Default: "./ErrorFiles/"

Depending on the training settings or your images features, you may also want to change other parameters in _config.py_ as the number of clusters to create.

The last script, _randomForest.py_, will create a new file in root directory, Results.csv. 

_RobustAutoencoder.py_ defines the architecture of the RAE models that are used to predict the label of the images.

#### Results output
The execution of _randomForest.py_ script will provide the results of PARDINUS on the test images. The file that store the results is called Results.csv. For each row, this file store the name of the image and the label assigned. Label 0 means that PARDINUS has classified the image as empty, while label 1 implies that there are animals within the image. The image name and the assigned label are separated by the symbol ";", following the CSV standars.

### Train new models using your own images

There are six scripts that you should execute to test trained models on your own images: _clustering.py, _applyClustering.py, autoencoders.py, applyAutoencoders.py, balanceErrorFile.py_ and _randomForest.py_, __in this order__. 

```
python clustering.py
python applyClustering.py
python autoencoders.py
python applyAutoencoders.py
python balanceErrorFile.py
python randomForest.py
```
The file _config.py_ contains variables that needs to be set to make the scripts work properly.

- IMAGE_FOLDER:  set where the clustered and equalized images for the trainig of the RAEs will be stored. Default: "./Data/"

- EMPTY_DATA: set where the empty images are stored. Default: IMAGE_FOLDER + "BBDDTrain/Empty"

- ANIMAL_DATA: set where the non-empty images are stored. Default: IMAGE_FOLDER + "BBDDTrain/Animal"

- TRAINED_MODELS_ROUTE: set where the trained models are stored. There should be one k-means clustering model, one random forest model and one RAE model for each cluster of images. Default: "./TrainedModels/"

- ERROR_FILES_ROUTE: set where the trained models are stored. Default: "./ErrorFiles/"

- ANIMAL_PROPORTION: set the proportion of animal images versus empty images (i.e 24 means that 24% of all images are non-empty)


Depending on the training settings or your images features, you may also want to change other parameters in config.py as the image width or height, the number of clusters or other training hyperparameters as number of epoch or batch size.

After run all the scripts, trained models should be stored at TRAINED_MODELS_ROUTE.



## Data Availability
The dataset used in this project is provided by WWF and is subject to restricted access. 

Researchers interested in using this dataset must request permission directly from WWF. The original photographs and data can be accessed via Wildlife Insights, project “Seguimiento Lince WWF España”. The details about the dataset and its metadata can be found at [this link](http://n2t.net/ark:/63614/w12001260). Note that this repository provides scripts and instructions to train and test models using your own datasets.
