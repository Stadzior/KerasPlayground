import os
import numpy as np

mask_suffix = "_Delmon_CompleteMM"

def get_filenames(path, extension, file_count, with_masks=False):
        filenames = list(map(lambda x: x.replace(".raw", ""), filter(lambda x: x.endswith(".raw"), os.listdir(path))))[:file_count]
        if with_masks:
                temp_filenames = []
                for filename in filenames:
                        temp_filenames.append(filename)
                        temp_filenames.append("{}{}".format(filename, mask_suffix))
                filenames = temp_filenames
        return filenames
        
def get_layer_ids(filenames, layer_count):
        ids = range(1, layer_count+1)
        return list(("{}_{}".format(filename, id) for filename in filenames for id in ids))

def load_data(image_path, image_datatype, mask_path, mask_datatype, layer_dims, file_count):
        data = [] 
        for filename in get_filenames(image_path, ".raw", file_count):
                print(filename)
                data += load_image_and_mask(filename, image_path, image_datatype, mask_path, mask_datatype, layer_dims)
        return data
        
def load_image(path, datatype, layer_dims, layer_count = None):   
        img_file = open(path, "rb")
        img_str = img_file.read()
        layer_size = layer_dims[0]*layer_dims[1]
        img_vector = np.fromstring(img_str, dtype=datatype)[:layer_size*layer_count] 
        layer_count = layer_count if layer_count != None else int(len(img_vector) / layer_size)
        return np.reshape(img_vector, (layer_count, layer_size))

def load_image_and_mask(filename, image_path, image_datatype, mask_path, mask_datatype, layer_dims, layer_count = None):
        images = load_image("{}/{}.raw".format(image_path, filename), image_datatype, layer_dims, layer_count)
        masks = load_image("{}/{}{}.raw".format(mask_path, filename, mask_suffix), mask_datatype, layer_dims, layer_count)
        return images + masks