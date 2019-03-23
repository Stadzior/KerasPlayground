from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import model
import dataProvider
import numpy as np

# batch_size = 128
# num_classes = 10
# epochs = 2

# # input image dimensions
# img_rows, img_cols = 512, 512

data = dataProvider.load_data("./images", np.int16, "./masks", np.int8, 5)
data2 = "hue"

# model = model.unet()

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])