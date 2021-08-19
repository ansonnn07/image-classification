# Image Classification with TensorFlow

# Summary
This project is built to teach you how to prepare your images and data for image classification task. There are three notebooks that can be referred in the root directory of this GitHub repository:
1. `Image collection.ipynb` -- collecting and organizing the images into folders
2. `Training - Sign Language.ipynb` -- preparing the dataset, building model from scratch, and training, on sign language recognition from gestures
3. `Training - Transfer Learning - Skin Cancer.ipynb` -- using pre-trained models for transfer learning, on skin cancer classification

These three notebooks contain very detailed explanations and comments to guide you through the code. Please refer to them for more details on all the steps.

This project also assumes that you already have some knowledge about deep learning with computer vision, especially the fundamental knowledge of convolutional neural networks. Otherwise, you would have a hard time understanding what is happening under the hood with the code in the notebooks.

You may also download the datasets used in this repository from the links below:
1. Sign Language (Gesture) Recognition: [Google Drive Link](https://drive.google.com/file/d/1EAcId2AJefByuUvDAL_6Ee5QdWo-ABSd/view?usp=sharing)
2. Skin Cancer Classification: [Kaggle Link](https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign)

# Package Installation
NOTE: It is assumed that you have installed Anaconda in your machine and know how to create virtual environments on your own.

You only need to install the required packages mentioned inside the `requirements.txt` file.
```
pip install -r requirements.txt
```
But to use TensorFlow with GPU support on your local machine, you will need to refer to this [YouTube video tutorial](https://youtu.be/hHWkvEcDBO0) for further instructions as this is a very error-prone process.