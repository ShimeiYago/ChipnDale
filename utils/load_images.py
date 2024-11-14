import numpy as np
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import ImageFile
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_images(dirpath, imglen):
    imglist = []
    file_names = []
    for i, path in enumerate(sorted(glob.glob(f"{dirpath}/*"))):
        img = load_img(path, grayscale=False, color_mode='rgb', target_size=(imglen, imglen))
        imglist.append(img_to_array(img))
        file_names.append(os.path.basename(path))

    return np.array(imglist), file_names
