#!/usr/bin/env python

import numpy as np
import os
from utils.models import CNN
from utils import load_images
from utils import GradCam
import argparse

DATASETS_DIR = 'datasets'

WORKSPACE = 'workspace'
INPUT_DIR = '01-classifier'
OUTDIR_NAME = '03-grad-cam'
TEST_A = 'testA'
TEST_B = 'testB'

IMGLEN = 150

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, help='dataset name')
    args = parser.parse_args()

    outdir = os.path.join(WORKSPACE, args.dataset_name, OUTDIR_NAME)
    os.makedirs(outdir, exist_ok=True)

    # load images
    dataset_dir = os.path.join(DATASETS_DIR, args.dataset_name)
    test_a_dir = os.path.join(dataset_dir, TEST_A)
    test_b_dir = os.path.join(dataset_dir, TEST_B)
    test_a_images = load_images(test_a_dir, IMGLEN)
    test_b_images = load_images(test_b_dir, IMGLEN)

    # normalize
    test_a_images /= 255
    test_b_images /= 255

    # load model
    model_weight_path = os.path.join(WORKSPACE, args.dataset_name, INPUT_DIR, 'weights.hdf5')
    model = CNN(IMGLEN)
    model.load_weights(model_weight_path)

    # calculate grad heatmap
    grad_cam = GradCam(model)

    chip_heatmaps = grad_cam(test_a_images)
    dale_heatmaps = grad_cam(test_b_images)

    # save
    outpath_a = os.path.join(outdir, f'{TEST_A}_heatmap.npy')
    outpath_b = os.path.join(outdir, f'{TEST_B}_heatmap.npy')
    np.save(outpath_a, chip_heatmaps)
    np.save(outpath_b, dale_heatmaps)


if __name__ == '__main__':
    main()
