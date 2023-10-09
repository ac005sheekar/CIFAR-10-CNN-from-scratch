###              Written by: Sheekar Banerjee             ####


#Dataset link: https://www.kaggle.com/datasets/sgazer/cifar10batchespy


import os
import numpy as np
import pandas as pd
#import pylab
from PIL import Image
#from IPython.display import SVG
import matplotlib.pyplot as plt
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)


'''
import seaborn as sns
sns.set(style="ticks", color_codes=True, font_scale=1.5)
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap
'''

import math
import timeit
from six.moves import cPickle as pickle
import platform
#from subprocess import check_output
import glob

import tensorflow as tf
import keras
#from keras.constraints import maxnorm
#from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
#from keras.utils.np_utils import to_categorical
#from keras.utils.vis_utils import model_to_dot
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from tqdm import tqdm_notebook
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Load and Prepare Data

def unpickle(fname):
    with open(fname, "rb") as f:
        result = pickle.load(f, encoding='bytes')
    return result


def getData():
    labels_training = []
    dataImgSet_training = []
    labels_test = []
    dataImgSet_test = []

    # use "data_batch_*" for just the training set
    for fname in glob.glob("C:/Users/SHEEKAR/PycharmProjects/scratch CNN Cifar10/data/cifar-10-batches-py/*data_batch*"):
        print("Getting data from:", fname)
        data = unpickle(fname)

        for i in range(10000):
            img_flat = data[b"data"][i]
            # fname = data[b"filenames"][i]
            labels_training.append(data[b"labels"][i])

            # consecutive 1024 entries store color channels of 32x32 image
            img_R = img_flat[0:1024].reshape((32, 32))
            img_G = img_flat[1024:2048].reshape((32, 32))
            img_B = img_flat[2048:3072].reshape((32, 32))

            imgFormat = np.array([img_R, img_G, img_B])
            imgFormat = np.transpose(imgFormat, (1, 2, 0))  # Change the shape 3,32,32 to 32,32,3
            dataImgSet_training.append(imgFormat)

    # use "test_batch_*" for just the test set
    for fname in glob.glob("C:/Users/SHEEKAR/PycharmProjects/scratch CNN Cifar10/data/cifar-10-batches-py/*test_batch*"):
        print("Getting data from:", fname)
        data = unpickle(fname)

        for i in range(10000):
            img_flat = data[b"data"][i]
            # fname = data[b"filenames"][i]
            labels_test.append(data[b"labels"][i])

            # consecutive 1024 entries store color channels of 32x32 image
            img_R = img_flat[0:1024].reshape((32, 32))
            img_G = img_flat[1024:2048].reshape((32, 32))
            img_B = img_flat[2048:3072].reshape((32, 32))

            imgFormat = np.array([img_R, img_G, img_B])
            imgFormat = np.transpose(imgFormat, (1, 2, 0))  # Change the shape 3,32,32 to 32,32,3
            dataImgSet_test.append(imgFormat)

    dataImgSet_training = np.array(dataImgSet_training)
    labels_training = np.array(labels_training)
    dataImgSet_test = np.array(dataImgSet_test)
    labels_test = np.array(labels_test)

    return dataImgSet_training, labels_training, dataImgSet_test, labels_test


X_train, y_train, X_test, y_test = getData()

labelNamesBytes = unpickle("C:/Users/SHEEKAR/PycharmProjects/scratch CNN Cifar10/data/cifar-10-batches-py/batches.meta")
labelNames = []
for name in labelNamesBytes[b'label_names']:
    labelNames.append(name.decode('ascii'))

labelNames = np.array(labelNames)

# fig = plt.figure(figsize=(6, 6))
# for i in range(0, 9):
#     ax = fig.add_subplot(330 + 1 + i)
#     plt.imshow(Image.fromarray(X_test[i]))
#     ax.set_title(labelNames[y_test[i]])
#     ax.axis('off')
#
# plt.show()


#Scaling
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

#Prepare the target variable
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

#Data Augmentation
datagen = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
    )
datagen.fit(X_train)

# Set Global Variables and Seed
num_classes = 10
input_shape = (32, 32, 3)
kernel = (3, 3)

# fix random seed for reproducibility
seed = 101
np.random.seed(seed)

#Create the CNN Model
model = Sequential()
model.add(Conv2D(64, kernel_size=kernel, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=kernel, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Visualize the network architecture (Only for Notebook)
#SVG(model_to_dot(model, show_shapes=True, show_layer_names=True, rankdir='TB').create(prog='dot', format='svg'))

#training
batch_size = 50
epochs = 75
lrate = 0.1
epsilon=1e-08
decay=1e-4
#optimizer = keras.optimizers.rmsprop(lr=lrate,decay=1e-4)
optimizer = keras.optimizers.Adadelta(lr=lrate ) #, epsilon=epsilon, decay=decay)
#optimizer = keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=epsilon, decay=decay)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=X_train.shape[0] // batch_size, epochs=epochs, verbose=1,
                    validation_data=(X_test,y_test))



#Plotting (Accuracy & Loss (training, validation))
# def plot_results(history):
#     acc = history.history['acc']
#     val_acc = history.history['val_acc']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#
#     epoch = range(epochs)
#
#     fig = plt.figure(figsize=(20,6))
#     ax1 = fig.add_subplot(121)
#     plt.plot(epoch, acc, 'b', label='Training acc')
#     plt.plot(epoch, val_acc, 'r', label='Validation acc')
#     ax1.set_title('Training and validation accuracy')
#     ax1.legend()
#
#     ax2 = fig.add_subplot(122)
#     plt.plot(epoch, loss, 'b', label='Training loss')
#     plt.plot(epoch, val_loss, 'r', label='Validation loss')
#     ax2.set_title('Training and validation loss')
#     ax2.legend()
#
#     plt.show()
#
# plot_results(history)
#
#
# #Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print("Test Accuracy: %.2f%%" % (scores[1]*100))
#
# #Saving the model
# model.save('cifar10_1')
#
# #Predict class of image in practice
#
# # How CNN Classifies an Image?
# img_idx = 122
# plt.imshow(X_test[img_idx],aspect='auto')
# print('Actual label:', labelNames[np.argmax(y_test[img_idx])])
# # Preper image to predict
# test_image =np.expand_dims(X_test[img_idx], axis=0)
# print('Input image shape:',test_image.shape)
# print('Predict Label:',labelNames[model.predict_classes(test_image,batch_size=1)[0]])
# print('\nPredict Probability:\n', model.predict_proba(test_image,batch_size=1))