#!/usr/bin/env python

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import os
from utils.models import model1 as CNN
from utils import load_images


INPUTDIR = 'datasets/training'
CHIPDIR = os.path.join(INPUTDIR, 'chip')
DALEDIR = os.path.join(INPUTDIR, 'dale')

OUTDIR = 'workspace/01-classifier'
os.makedirs(OUTDIR, exist_ok=True)

IMGLEN = 150

EPOCHS = 5
PATIENCE = 20


def main():
    # prepare_dataset
    x_train, x_test, y_train, y_test = prepare_dataset(0.2)

    print('train shape:', x_train.shape, y_train.shape)
    print('test shape:', x_test.shape, y_test.shape)

    # learning
    model = CNN(IMGLEN)

    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0, mode='auto')

    history = model.fit(x_train, y_train,
                        batch_size=10,
                        epochs=EPOCHS,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[early_stop])

    # save history
    histdf = pd.DataFrame(history.history)
    with open(os.path.join(OUTDIR, 'history.csv'), 'w') as f:
        histdf.to_csv(f)

    # save model wights
    model.save_weights(os.path.join(OUTDIR, 'weights.hdf5'))


def prepare_dataset(test_size=0.2):
    # load
    chip_images = load_images(CHIPDIR, IMGLEN)
    dale_images = load_images(DALEDIR, IMGLEN)

    N_test = int(chip_images.shape[0]*test_size)
    N_train = chip_images.shape[0] - N_test

    x_train = np.append(chip_images[:N_train], dale_images[:N_train], axis=0)
    y_train = np.array([0]*N_train + [1]*N_train)
    x_test = np.append(chip_images[N_train:], dale_images[N_train:], axis=0)
    y_test = np.array([0]*N_test + [1]*N_test)

    # normalize
    x_train /= 255
    x_test /= 255

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    main()
