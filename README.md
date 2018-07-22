## Deep_learning_CNN
## Convolution Neural Network Code

## Problem Statement:
Problem statement is classify image of cat or dog.

Data set have image of cat and dog. I have build CNN model to classify them. I have shown very simple way to build CNN using keras.

## Model achieved accuracy of :
Traning accuracy = 98%
Test accuracy = 88%
These accuracy can be increased by using better input_shape and by adding more convolution layer as shown below.

## Trial 1
Convolutional Layer: 2,
input_shape = (64,64,3),
No Dropout rate added,
Accuracy on test set = 78%

## Trial 2
Convolutional Layer: 3,
input_shape = (128,128,3),
Added Dropout for avoiding over fitting. Dropout rate = 0.5,
With these changes acc of test set has being increased to 88%

## Single Prediction:
I have also show how to Making single prediction, that given a single image model to predict image is of cat or dog ??


