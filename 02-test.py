#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
import argparse
import math

from tensorflow.keras.preprocessing.image import array_to_img
from utils import load_images
from utils.models import CNN


DATASETS_DIR = 'datasets'

WORKSPACE = 'workspace'
INPUT_DIR = '01-classifier'
OUTDIR_NAME = '02-test'
TEST_A = 'testA'
TEST_B = 'testB'

IMGLEN = 150

N_PLT_COLUMNS = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, help='dataset name')
    args = parser.parse_args()

    dataset_dir = os.path.join(DATASETS_DIR, args.dataset_name)
    outdir = os.path.join(WORKSPACE, args.dataset_name, OUTDIR_NAME)
    os.makedirs(outdir, exist_ok=True)

    test_a_dir = os.path.join(dataset_dir, TEST_A)
    test_b_dir = os.path.join(dataset_dir, TEST_B)
    
    test_a_images = load_images(test_a_dir, IMGLEN)
    test_b_images = load_images(test_b_dir, IMGLEN)

    # normalize
    test_a_images /= 255
    test_b_images /= 255

    # load modal
    model_weight_path = os.path.join(WORKSPACE, args.dataset_name, INPUT_DIR, 'weights.hdf5')
    model = CNN(IMGLEN)
    model.load_weights(model_weight_path)

    # predict A
    outpath_a = os.path.join(outdir, f'{TEST_A}.png')
    test_pred(model, test_a_images, outpath_a, 'A')

    # predict B
    outpath_b = os.path.join(outdir, f'{TEST_B}.png')
    test_pred(model, test_b_images, outpath_b, 'B')


def test_pred(model, images, outpath, mode):
    preds = model.predict(images)

    n_rows = math.ceil(len(preds) / N_PLT_COLUMNS)
    
    plt.figure(figsize=(3*N_PLT_COLUMNS, 4*n_rows))
    for i, pred in enumerate(preds):
        plt.subplot(n_rows, N_PLT_COLUMNS, i+1)
        if(pred[0]>pred[1] and mode=='A' or pred[0]<pred[1] and mode=='B'):
            ox = 'o'
        else:
            ox = 'x'
        plt.title(f'{ox} ({pred[0]:.2f})')
        plt.imshow(array_to_img(images[i]))

        plt.savefig(outpath)


if __name__ == '__main__':
    main()
