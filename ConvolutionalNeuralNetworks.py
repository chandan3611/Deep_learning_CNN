# Build Convolutional Neural Networks

#Import Keras package.
# Sequential pakage is used to initilize our neural network
from keras.models import Sequential
from keras.layers import Dropout

'''
# Conv2D is used to make first step in CNN that is adding convolutional layers.
# Since we are working on image, and images are 2D unlike video which are 3D(3rd dimention is time).
# So we will w=be using Conv2D pakage. 
'''
from keras.layers import Conv2D

#MaxPooling : Build step 2 that is pooling step
from keras.layers import MaxPooling2D

'''
# Flatten : Used to build step 3, that is converting all pooled feature maps that we have created through
# convolution and max pooling into this large feature vector and that will become input layer to fully connected layer.
'''
from keras.layers import Flatten

# Dense : Used to add fully connected layer in artifical neural network
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 : Convolution
'''
# Here order of input_shape parameter is important. Since we are using tensorflow backend the order is 2D array dimension followed by number od channels.
# And it will be otherwise in theno backend. Number of channel = 3 for color image and 2 for black n white channel
# Next important parameter is activation function, we are using "relu" activation function as we dont want any negative pixel value in feature maps.
# We need to remove these negative pixels in order to have non linearity in our CNN.
'''
classifier.add(Conv2D(32, 3, 3, input_shape = (128,128,3), activation = 'relu'))

# Step 2: Pooling
'''
# Poolong step will reduce the size of feature map. We take 2X2 sub table and slide over feature map and each time we take max of four cells.
# Taking max is called max pooling, and we slide this sub table with stride of 2 not with stride of 1. Which will result in reduced size feature map.
# Size of feature map is divided by 2. We apply max pooling on all the feature maps and this will obtain next layer called pooling layer.
'''
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding a 3rd convolutional layer
classifier.add(Conv2D(64, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step 3: Flattening
'''
# Taking all pooled feature maps and put them into a single vector. This vector will be huge.
# This single vector will be intput to ANN layer.
# 1) Why dont we lose special structure by flattering these feature map: That is because by creating feature maps we extracted the special structure information by getting high numbers in feature maps.
# These high numbers represents special structure in image and these high numbers preserved by convotutiona and max pooling step.
# 2) Why dont we take all pixels and flatten them without appling previous steps ? : If we do so then each pixel will represent one pixel independently. So we get information about only 
# one pixel not that how this pixel is connected to other pixels around it. So we dont and any information about special structure around this pixel.
'''
classifier.add(Flatten())

# Step 4: Full Connection
'''
Builing classic ANN composed of fully connected layer. In previous step we have created huge single vector that can be used as
input layer of ANN. Because ANN can be a good classifier of non linear problem.
'''
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(rate=0.5))
classifier.add(Dense(units=1, activation='sigmoid'))

# Step 5 : Compile CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 6: Fitting image to CNN
'''
We need to perform image augmentation step that basically means image pre-processing to prevent over fitting. And that can be done by ImageDataGenerator().
In case of images model need to lots of images to find and generalize some correlation between the images. And that is where image augmentation is used.
Image augmentation trick will create many batches of image and in each batch it will apply some random transformation on randomally selected images like roatation, filliping, shifting, shearing them etc..
Due to this transformation model will get many diverse images inside these batches, therefore lot more material to train. The transformation are random so model will not find same image across the batches.
'''
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('C:\\Users\\Chandan.S\\Desktop\\DeepLearning\\CNN\\dataset\\training_set',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('C:\\Users\\Chandan.S\\Desktop\\DeepLearning\\CNN\\dataset\\test_set',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)

#-------------------------------------------------------------------------------------------------------
'''
Our CNN model is ready we see that model is having accuracy of 88% of test set. Moving further I will show one application of this CNN model. In which we will try to predict one single image. Wheather it is cat or dog.
We have two images one of cat and other one of dog. And same we will be predicting using this model.
Data present at dataset\single_prediction
'''

# Making single prediction
#Import numpy for pre-processing the image so that it can be excepted  by predict method
import numpy as np
from keras.preprocessing import image

# Load the impage on which we want to do the prediction.
'''
load_img : Funtion used to load the image. This function require 2 argument.
1st argument : Path of the image.
2nd argument : targest side, that must be same as we have used in training. Here we have used 128X128 image for training so 
predict funtcion will also expect the same size.
'''
test_image = image.load_img('C:\\Users\\Chandan.S\\Desktop\\DeepLearning\\CNN\\dataset\\single_prediction\\cat_or_dog_1.jpg',target_size=(128, 128)) 

'''
After importing we can see that the size and type of test_image as shown below.
test_image:
    type = Image
    Size = (128, 128)
'''

# Adding 3rd dimension
'''
We see that test_image is 2 dimensional, here we need to add one more dimension.
That is because input layer of CNN has 3 dimension, input_shape = (128,128,3). So we need to 3rd dimention to 
test_image and that is for color. This can be done by img_to_array()
'''
test_image = image.img_to_array(test_image)

'''
After adding 3rd dimension test_image will become as:
test_image:
    type = Image
    Size = (128, 128, 3) : Now test_image became 3 dimensional of size (128,128,3)
    same as CNN input layer.
'''

# Add 4th dimension
'''
THis 4th dimension crossponds to batch. Function of neural network cannot accept single input image. It only accepts inputs in a batch, even if batch contains only one image.
Here we will have 1 batch on one input but we can have several batches of several inputs.
expand_dims() : This function takes two images. 1st argument = test_image and 2nd argument is axis which represent index of new dimention.
'''
test_image = np.expand_dims(test_image, axis=0)

'''
After adding 4th dimension test_image will become as:
test_image:
    type = Image
    Size = (1,128, 128, 3) : Now test_image became 3 dimensional of size (128,128,3)
    same as CNN input layer.
'''

# Prediction
result = classifier.predict(test_image)

'''
classifier.predict(test_image)
Out[41]: array([[1.]], dtype=float32)
Here result of prediction is 1. Need to figure out what 1 crossponds, Cat or dog.
'''

# To know mapping
'''
To know the mapping we will use class_indices attribute that will tell us the mapping between
string cat and dog and their numeric value. 
'''
training_set.class_indices

'''
training_set.class_indices
Out[43]: {'cats': 0, 'dogs': 1}
'''

# So here we see that our model prediction is correct. Model has predicted "1" and that crossponds to dog.
if result[0][0] == 1:
    prediction = 'Dog'
else:
    prediction = 'Cat'


