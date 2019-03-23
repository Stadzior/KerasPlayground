import os
import numpy as np

layer_dimensions = (512, 512)
layer_size = layer_dimensions[0]*layer_dimensions[1]
mask_suffix = "_Delmon_CompleteMM"

def load_data(image_path, image_datatype, mask_path, mask_datatype, file_count):
        data = []
        for filename in list(map(lambda x: x.replace(".raw", ""), filter(lambda x: x.endswith(".raw"), os.listdir(image_path))))[:file_count]:  
                print(filename)
                data += load_file(filename, image_path, image_datatype, mask_path, mask_datatype)
        return data
        
def load_image(path, datatype):   
        img_file = open(path, "rb")
        img_str = img_file.read()
        img_vector = np.fromstring(img_str, dtype=datatype) 
        layers_count = int(len(img_vector) / layer_size)
        return np.reshape(img_vector, (layers_count, layer_size))

def load_file(filename, image_path, image_datatype, mask_path, mask_datatype):
        images = load_image("{}/{}.raw".format(image_path, filename), image_datatype)
        masks = load_image("{}/{}{}.raw".format(mask_path, filename, mask_suffix), mask_datatype)
        img_mask_tuples = list(zip(images, masks))
        return img_mask_tuples