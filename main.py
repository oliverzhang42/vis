"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""

from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping

import numpy as np
import resnet
import os


batch_size = 32
nb_classes = 2
nb_epoch = 50
data_augmentation = False

# input image dimensions
img_rows, img_cols = 224, 224

img_channels = 3

model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
# model.summary()
# model = InceptionResNetV2()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.load_weights('plot_resnet_weights.22-0.99.hdf5')

model.save("resnet_ppg")