#!/usr/bin/env python

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import glob
import argparse

DATASETS_DIR = 'datasets'

WORKSPACE = 'workspace'
INPUT_DIR = '03-grad-cam'
TEST_A_HEATMAP = "testA_heatmap.npy"
TEST_B_HEATMAP = "testB_heatmap.npy"
OUTDIR_NAME = '04-heatmaps'
TEST_A = 'testA'
TEST_B = 'testB'

INTENSITY = 0.9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, help='dataset name')
    args = parser.parse_args()

    outdir = os.path.join(WORKSPACE, args.dataset_name, OUTDIR_NAME)
    outdir_a = os.path.join(outdir, TEST_A)
    outdir_b = os.path.join(outdir, TEST_B)
    os.makedirs(outdir_a, exist_ok=True)
    os.makedirs(outdir_b, exist_ok=True)

    dataset_dir = os.path.join(DATASETS_DIR, args.dataset_name)
    test_a_dir = os.path.join(dataset_dir, TEST_A)
    test_b_dir = os.path.join(dataset_dir, TEST_B)

    inputdir = os.path.join(WORKSPACE, args.dataset_name, INPUT_DIR)
    test_a_heatmaps_path = os.path.join(inputdir, TEST_A_HEATMAP)
    test_b_heatmaps_path = os.path.join(inputdir, TEST_B_HEATMAP)
    test_a_heatmaps = np.load(test_a_heatmaps_path)
    test_b_heatmaps = np.load(test_b_heatmaps_path)

    # output testA
    joined_imgs = join_imgs_heatmaps(test_a_dir, test_a_heatmaps)
    output_imgs(joined_imgs, outdir_a)

    # output testB
    joined_imgs = join_imgs_heatmaps(test_b_dir, test_b_heatmaps)
    output_imgs(joined_imgs, outdir_b)


def join_imgs_heatmaps(imgdir, heatmaps):
    imgs = []
    for i, path in enumerate(sorted(glob.glob(f"{imgdir}/*"))):
        img = cv2.imread(path)
        heatmap = cv2.resize(heatmaps[i], (img.shape[1], img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

        # heatmap = heatmap
        heatmap = np.clip(heatmap.astype(np.int16), 0, 255)

        img = heatmap * INTENSITY + img
        img = np.clip(img.astype(np.int16), 0, 255)

        imgs.append(img)

    return imgs


def output_imgs(imgs_list, outdir):
    for i, img in enumerate(imgs_list):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.matshow(img)

        ax.axis("off")

        filename = f'{i:02d}.png'
        plt.savefig(os.path.join(outdir, filename), bbox_inches="tight")
        plt.close()


if __name__ == '__main__':
    main()
