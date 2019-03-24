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

image_layer_count = 469
batch_size = 469*2 # one batch contains one image and one mask
class_count = 2 # image = 0 and mask = 1
train_test_ratio = 0.8 # 8 out of 10 images would be used for training
epochs = 1

# Datasets
ids = dataProvider.get_layer_ids("./images", ".raw", 1, image_layer_count)
random.shuffle(ids)
labels = {id: 1 if dataProvider.mask_suffix in id else 0 for id in ids}
train_samples_count = int(batch_size*train_test_ratio)
train_ids = ids[:train_samples_count]
test_ids = ids[train_samples_count:]

# Generators
train_generator = dataGenerator.ImageVsMaskDataGenerator(train_ids, batch_size, labels)
test_generator = dataGenerator.ImageVsMaskDataGenerator(test_ids, batch_size, labels)

# Design model
model = model.unet()

# Train model on dataset
model.fit_generator(validgenerator=train_generator,
                    use_multiprocessing=True,
                    epochs=epochs,
                    workers=6)

score = model.evaluate_generator(test_generator,
                                 use_multiprocessing=True,
                                epochs=epochs,
                                workers=6)
print('Test loss:', score[0])
print('Test accuracy:', score[1])