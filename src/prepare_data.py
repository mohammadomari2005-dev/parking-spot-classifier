import os
import numpy as np
from skimage.io import imread 
from skimage.transform import resize

import sys
import os
sys.path.append(os.path.dirname(__file__))
from config import *


def load_data():
    data = []
    labels = []

    for category_idx, category in enumerate(CATEGORIES):
        for file in os.listdir(os.path.join(INPUT_DIR, category)):
            img_path = os.path.join(INPUT_DIR, category, file)
            img = imread(img_path)
            img = img[:, :, :3]
            img = resize(img, IMAGE_SIZE)
            img = img / 255.0

            print('Sample image shape:', img.shape)
            data.append(img.flatten())
            labels.append(category_idx)

    data = np.asarray(data)
    labels = np.asarray(labels)

    return data, labels
