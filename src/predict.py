import pickle
import argparse
from skimage.io import imread 
from skimage.transform import resize 
import numpy as np
from config import *


def predict(image_path):
    print('Loading model...')
    model = pickle.load(open(MODEL_PATH, 'rb'))
    print('Model loaded!')

    img = imread(image_path)
    img = img[:, :, :3]
    print('Image loaded!')

    img = resize(img, IMAGE_SIZE)
    img = img / 255.0
    print('Image shape:', img.shape)
    img = img.flatten().reshape(1, -1)
    prediction = model.predict(img)
    print('Prediction: {}'.format(CATEGORIES[prediction[0]]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    args = parser.parse_args()
    predict(args.image_path)