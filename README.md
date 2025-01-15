# Face-Mask-Detection-Using-CNN

A deep learning project to detect whether a person is wearing a face mask or not, developed using Convolutional Neural Networks (CNN). The model achieves an accuracy of 96% on the test set.

# Features

Detects the presence or absence of a face mask in images and video streams.
Can be integrated with real-time video feeds for live mask detection.
Built with Python and TensorFlow/Keras.


# Dataset

The model was trained on a publicly available dataset containing images of people with and without face masks.
Images were preprocessed and augmented to improve model performance and generalization.


# Model Architecture

The model architecture consists of the following layers:

1. Convolutional Layers:

First convolutional layer with 32 filters, kernel size (3, 3), ReLU activation, and same padding.
Second convolutional layer with 64 filters, kernel size (3, 3), and ReLU activation

2. Pooling Layers:

Max pooling layers with a pool size of (2, 2) following each convolutional layer.

3. Fully Connected Layers:

A flattening layer followed by a dense layer with 128 neurons and ReLU activation.
Dropout layers with a rate of 0.5 to prevent overfitting.
Another dense layer with 64 neurons and ReLU activation.

4. Output Layer:

Dense layer with 2 neurons and softmax activation for binary classification (mask/no mask).


# Accuracy

The model achieved 96% accuracy on the test set.

# Prerequisites

Python 3.8+
TensorFlow/Keras
OpenCV
NumPy
Matplotlib

# Summary
This project implements a face mask detection system using Convolutional Neural Networks (CNN). The architecture includes convolutional layers for feature extraction, max pooling for dimensionality reduction, fully connected layers for classification, and dropout layers to prevent overfitting. The model, trained on a dataset of masked and unmasked face images, achieved 96% accuracy in classifying images
