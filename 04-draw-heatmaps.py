#!/usr/bin/env python

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import glob


INPUTDIR = 'workspace/03-grad-cam'
CHIP_HEATMAPS_PATH = os.path.join(INPUTDIR, 'chip_heatmaps.npy')
DALE_HEATMAPS_PATH = os.path.join(INPUTDIR, 'dale_heatmaps.npy')

IMGDIR = 'datasets/test'
CHIP_IMGS_DIR = os.path.join(IMGDIR, 'chip')
DALE_IMGS_DIR = os.path.join(IMGDIR, 'dale')

OUTDIR = 'workspace/04-heatmaps'
OUTDIR_CHIP = os.path.join(OUTDIR, 'chip')
OUTDIR_DALE = os.path.join(OUTDIR, 'dale')
os.makedirs(OUTDIR_CHIP, exist_ok=True)
os.makedirs(OUTDIR_DALE, exist_ok=True)

INTENSITY = 0.9


def main():
    chip_heatmaps = np.load(CHIP_HEATMAPS_PATH)
    joined_imgs = join_imgs_heatmaps(CHIP_IMGS_DIR, chip_heatmaps)
    output_imgs(joined_imgs, OUTDIR_CHIP)

    dale_heatmaps = np.load(DALE_HEATMAPS_PATH)
    joined_imgs = join_imgs_heatmaps(DALE_IMGS_DIR, dale_heatmaps)
    output_imgs(joined_imgs, OUTDIR_DALE)


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
