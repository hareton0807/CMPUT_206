'''
Train a simple deep CNN on the CIFAR10 small images dataset.
It achieves 71% validation accuracy in 20 epochs with RGB input and normalization. (Can you get better?)
Hints: Batch Normalization, Add more Con2D layers and make sure of the input and output sizes, Different activation functions.
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

batch_size = 32
num_classes = 10
epochs = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model_color.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

'''
##############################################################
#           WRITE YOUR CODE HERE                             #
# Convert data to grey images instead of RGB                 #
# and put the results back into x_train and x_test           #
# The result should be of size nsamples x height x width x 1 #
##############################################################
'''
def rgb2gray(rgb):
  gray = np.dot(rgb[...,:3],[0.299,0.587,0.114])
  return gray.reshape(32,32,1)

nsamples,height,width,_ = x_train.shape
x_train_temp = np.zeros((nsamples,height,width,1),dtype = float)
for i in range(nsamples):
  result = rgb2gray(x_train[i])    
  x_train_temp[i] = result
x_train = x_train_temp

nsamples,height,width,_ = x_test.shape
x_test_temp = np.zeros((nsamples,height,width,1),dtype = float)
for i in range(nsamples):
  result = rgb2gray(x_test[i])
  x_test[i] = result
x_test = x_test_temp

print(x_train[1])
print('x_train.shape:',x_train.shape)
print('x_test.shape:',x_test.shape)

  
# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        if x_train[idx].shape[-1] == 3:
            plt.imshow(x_train[idx].squeeze().astype('uint8')) #change cmap='gray' when using gray input
        else:
            plt.imshow(x_train[idx].squeeze().astype('uint8'),cmap='gray')

        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate Stochastic Gradient Descend optimizer
opt = keras.optimizers.SGD()

# Let's train the model
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

'''
#########################################################
#                 WRITE YOUR CODE HERE                  #
# Normalize image values from -0.5:0.5 instead of 0:225 #
#########################################################
'''
#Use opencv nomralize funtion
for i in range(x_train.shape[0]):
  x_train[i] = cv2.normalize(x_train[i],x_train[i],alpha = -0.5, beta = 0.5)
  
for i in range(x_test.shape[0]):
  x_test[i] = cv2.normalize(x_test[i],x_test[i],alpha = -0.5, beta = 0.5)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

