import numpy as np
import keras
import dataProvider

class ImageVsMaskDataGenerator(keras.utils.Sequence):
    "Generates data for Keras"
    def __init__(self, filenames, layer_dims, layer_count, class_count=2, shuffle=True):
        "Initialization"
        self.filenames = filenames
        self.class_count = class_count
        self.layer_count = layer_count
        self.layer_dims = layer_dims
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        "Generate one batch of data"        
        # Generate data
        X, y = self.__data_generation(self.filenames[index])
        return X, y

    def __data_generation(self, filename):
        "Generates data containing one image and mask samples"        
        layer_size = self.layer_dims[0]*self.layer_dims[1]
        images = dataProvider.load_image(".images/{}.raw".format(filename), np.dtype("int16"), self.layer_dims, self.layer_count)
        masks = dataProvider.load_image(".masks/{}{}.raw".format(filename, dataProvider.mask_suffix), np.dtype("int8"), self.layer_dims, self.layer_count)
        dataset_size = len(images)+len(masks)
        X = np.empty((dataset_size, layer_size))
        y = np.empty(dataset_size, dtype=int)

        for i, sample in enumerate(images):
            X[i,] = sample
            y[i] = 0
        
        for i, sample in enumerate(masks, len(images)):
            X[i,] = sample
            y[i] = 1
            
        return X, keras.utils.to_categorical(y, num_classes=self.class_count)