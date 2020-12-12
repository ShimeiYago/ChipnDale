#!/usr/bin/env python

import numpy as np
import os
from utils.models import model1 as CNN
from utils import load_images
from utils import GradCam

INPUTDIR = 'datasets/test'
CHIPDIR = os.path.join(INPUTDIR, 'chip')
DALEDIR = os.path.join(INPUTDIR, 'dale')

OUTDIR = 'workspace/03-grad-cam'
os.makedirs(OUTDIR, exist_ok=True)
OUTPATH_CHIP = os.path.join(OUTDIR, 'chip_heatmaps.npy')
OUTPATH_DALE = os.path.join(OUTDIR, 'dale_heatmaps.npy')

MODEL_WEIGHT_PATH = 'workspace/01-classifier/weights.hdf5'

IMGLEN = 150

LAST_CONV_LAYER_NAME = 'conv2d_2'


def main():
    # load images
    chip_images = load_images(CHIPDIR, IMGLEN)
    dale_images = load_images(DALEDIR, IMGLEN)

    # normalize
    chip_images /= 255
    dale_images /= 255

    # load model
    model = CNN(IMGLEN)
    model.load_weights(MODEL_WEIGHT_PATH)

    # calculate grad heatmap
    grad_cam = GradCam(model, LAST_CONV_LAYER_NAME)

    chip_heatmaps = grad_cam(chip_images)
    dale_heatmaps = grad_cam(dale_images)

    # save
    np.save(OUTPATH_CHIP, chip_heatmaps)
    np.save(OUTPATH_DALE, dale_heatmaps)


if __name__ == '__main__':
    main()
