# Weakly supervised discarding of photo-trapping empty images based on autoencoders

This repository is the official implementation of the paper “Weakly supervised discarding of photo-trapping empty images based on autoencoders”. PARDINUS, built on the foundation of weakly-supervised learning, is the proposed tool to automatically detect empty or blank images.


## Install

Install requirements.txt using the following python code in a clean anaconda environment (Python 3.9 is reccomended):

```
pip install -r requirements.txt

```


## Code organization and usage
The code is organized into two main folders: _Train_ and _Test_. _Test_ folder contains the scripts to test or evaluate pretrained models on your own images. _Train_ folder contains the scripts for training your own models.

### Test or inference

There are three scripts that you should execute to test trained models on your own images: _clustering.py, autoencoders.py_ and _randomForest.py_, __in this order__. 

```
python clustering.py
python autoencoders.py
python randomForest.py
```


The file _config.py_ contains variables that needs to be set to make the scripts work properly.

- IMAGE_FOLDER: set where the images are stored. This folder will also contain clustered and equalized images. Default: "./Data/"
- TEST_IMAGES: set where the original images, your own images, are stored. Default: "./Data/"
- TRAINED_MODELS_ROUTE: set where the trained models are stored. There should be one k-means clustering model, one random forest model and one RAE model for each cluster of images. Default: "./TrainedModels/"
- ERROR_FILES_ROUTE: set where the trained models are stored. Default: "./ErrorFiles/"

Depending on the training settings or your images features, you may also want to change other parameters in _config.py_ as the image width or height or the number of clusters.

The last script, _randomForest.py_, will create a new file in root directory, Results.txt. This file store, for each row, the name of the image and the label assigned. Label 0 means that PARDINUS has classified the image as empty, while label 1 implies that there are animals within the image.

### Train new models using your own images




