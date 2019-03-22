import os
import numpy as np

layer_dimensions = (512, 512)
layer_size = layer_dimensions[0]*layer_dimensions[1]
mask_suffix = "_Delmon_CompleteMM"
img_mask_isLung = []

def load_data(images_path, images_datatype, masks_path, masks_datatype):
    for filename in filter(lambda x: x.endswith(".raw"), os.listdir(images_path)):        
        img_layered = load_image('{}/{}'.format(images_path, filename), images_datatype)
        mask_layered = load_image('{}/{}')


def load_image(path, datatype):   
    img_file = open(path)
    img_str = img_file.read()
    img_vector = np.fromstring(img_str, dtype=datatype) 
    layers_count = int(len(img_vector) / layer_size)
    return np.reshape(img_vector, (layers_count, layer_size))

def load_images(path, datatype):
    for file in filter(lambda x: x.endswith(".raw"), os.listdir()):
        f = open(file, 'rb')
        img_str = f.read()
            
        # converting to a int16 numpy array
        ct_image_as_vector = np.fromstring(img_str, dtype=np.int8) if "MM" in file else np.fromstring(img_str, np.int16)

        #adding to 



def load_masks():
    for file in filter(lambda x: x.endswith(".raw"), os.listdir("./masks")):
