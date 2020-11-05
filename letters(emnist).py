'''##################Used to allocate GPU memory to tensorflow########################'''
'''
#Run before starting the program
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
'''
'''##################################################################################'''

"""##############################################
'''## STEP 1 ##'''
#Importing dataset
import pandas as pd

dataset = pd.read_csv('C:/Users/daryl/Desktop/ip/emnist/emnist-byclass-train.csv', header = None)
X_train = dataset.iloc[0:, 1:].values
y_train = dataset.iloc[0:, :1].values


dataset = pd.read_csv('C:/Users/daryl/Desktop/ip/emnist/emnist-byclass-test.csv', header = None)
X_test = dataset.iloc[:, 1:].values
y_test = dataset.iloc[:, :1].values

'''## STEP 2 ##'''
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# convert class vectors to binary class matrices
import keras
num_classes = 62
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

hist = classifier.fit(X_train, y_train, batch_size = 128, epochs = 10, verbose = 1, validation_data = (X_test, y_test))
score = classifier.evaluate(X_test, y_test, verbose = 0)
###############################################"""


'''## STEP 3 ##'''
#Importing libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout

#Initializing the CNN
classifier = Sequential()

#Step1 - Convolution
#No of feature maps, rows, columns in feature detector, input_shape format is different for tensoflow and Theano
classifier.add(Convolution2D(32, (3, 3), input_shape = (128, 128, 1), activation = 'relu'))
#Step2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))

#Adding another convolutional layer to impporve accuracy
classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))

classifier.add(Convolution2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))

classifier.add(Convolution2D(256, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))

#Step3 - Flattening
classifier.add(Flatten())

#Step4 - Full Connection
#Adding the hidden layer
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dropout(0.2))

#Adding output layer
classifier.add(Dense(52, activation = 'softmax'))

#Conpiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Taken from Keras Documentation#################
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.3,
        zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

import os
len_train = sum([len(files) for r, d, files in os.walk("D:/ML Project/Dataset/train_set/")])
len_test = sum([len(files) for r, d, files in os.walk("D:/ML Project/Dataset/test_set/")])

training_set = train_datagen.flow_from_directory(
        'D:/ML Project/Dataset/train_set',
        target_size=(128, 128),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'D:/ML Project/Dataset/test_set',
        target_size=(128, 128),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical')

classifier.fit_generator(
        training_set,
        steps_per_epoch=len_train/64,
        epochs=25,
        validation_data=test_set,
        validation_steps=len_test/64)


classifier.save('D:/ML Project/letter(only).h5')