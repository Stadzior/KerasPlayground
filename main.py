from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import model
import dataProvider
import dataGenerator
import numpy as np
import random

layer_dims = (512,512)
layer_count = 5
class_count = 2 # image = 0 and mask = 1
train_test_ratio = 0.8 # 8 out of 10 files would be used for training
file_count = 3 # load n files (image+mask pairs)
epochs = 1

# Datasets
filenames = dataProvider.get_filenames("./images", ".raw", file_count, True)
train_file_count = int(file_count*train_test_ratio)
train_filenames = filenames[:train_file_count]
test_filenames = filenames[train_file_count:]

# Generators
train_generator = dataGenerator.DataGenerator(train_filenames, layer_dims, layer_count)
test_generator = dataGenerator.DataGenerator(test_filenames, layer_dims, layer_count)

# Design model
model = model.unet()

# Train model on dataset
model.fit_generator(generator=train_generator,
                    use_multiprocessing=True,
                    epochs=epochs,
                    workers=6)

score = model.evaluate_generator(generator=test_generator,
                                 use_multiprocessing=True,
                                 workers=6)
print('Test loss:', score[0])
print('Test accuracy:', score[1])