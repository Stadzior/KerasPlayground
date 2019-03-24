from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import model
import dataProvider
import numpy as np

image_layer_count = 469
batch_size = 469*2 # one batch contains one image and one mask
class_count = 2 # image and mask
train_test_ratio = 0.8 # 8 out of 10 images would be used for training
epochs = 2

# Datasets
ids = dataProvider.get_layer_ids("./images", ".raw", 1, image_layer_count)
print(ids)
#labels = # Labels

# # Generators self, filenames, batch_size, labels, class_count=2, shuffle=True
# training_generator = DataGenerator(partition['train'], labels, **params)
# validation_generator = DataGenerator(partition['validation'], labels, **params)

# # Design model
# model = Sequential()
# [...] # Architecture
# model.compile()

# # Train model on dataset
# model.fit_generator(generator=training_generator,
#                     validation_data=validation_generator,
#                     use_multiprocessing=True,
#                     workers=6)

# data = dataProvider.load_data("./images", np.int16, "./masks", np.int8, 5)
# data2 = "hue"

# model = model.unet()

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])