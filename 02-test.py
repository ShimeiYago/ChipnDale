#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
import argparse
import math
import csv

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
    
    test_a_images, test_a_names = load_images(test_a_dir, IMGLEN)
    test_b_images, test_b_names = load_images(test_b_dir, IMGLEN)

    # normalize
    test_a_images /= 255
    test_b_images /= 255

    # load modal
    model_weight_path = os.path.join(WORKSPACE, args.dataset_name, INPUT_DIR, 'weights.hdf5')
    model = CNN(IMGLEN)
    model.load_weights(model_weight_path)

    # predict A
    test_pred(model, test_a_images, test_a_names, outdir, TEST_A, 'A')

    # predict B
    test_pred(model, test_b_images, test_b_names, outdir, TEST_B, 'B')


def test_pred(model, images, file_names, outdir, out_file_name, mode):
    preds = model.predict(images)
        
    if mode=="A": dataset_index = 0
    else: dataset_index = 1
    
    # output result csv
    header = ["file", "probability"]
    results = [[file_names[i], preds[i][dataset_index]] for i in range(len(preds))]
    csv_outpath = os.path.join(outdir, f'results-{out_file_name}.csv')
    with open(csv_outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)
        
    # output image
    img_outpath = os.path.join(outdir, f'failed-{out_file_name}.png')
    failed_index_list = [i for i, pred in enumerate(preds) if pred[dataset_index] <= 0.5]        
    n_rows = math.ceil(len(failed_index_list) / N_PLT_COLUMNS)
    plt.figure(figsize=(3*N_PLT_COLUMNS, 4*n_rows))
    for i, failed_index in enumerate(failed_index_list):
        pred = preds[failed_index]
        plt.subplot(n_rows, N_PLT_COLUMNS, i+1)
        plt.title(f'{pred[dataset_index]:.2f}')
        plt.imshow(array_to_img(images[failed_index]))

        plt.savefig(img_outpath)

    if (len(failed_index_list) == 0):
        print(f"No failed for {out_file_name}")
        


if __name__ == '__main__':
    main()
