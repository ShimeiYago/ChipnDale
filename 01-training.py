#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse
from tensorflow.keras.callbacks import EarlyStopping
import os
from utils.models import model1 as CNN
from utils import load_images


DATASETS_DIR = 'datasets'
TRAIN_A = 'trainA'
TRAIN_B = 'trainB'
TEST_A = 'testA'
TEST_B = 'testB'

IMGLEN = 150

WORKSPACE = 'workspace'
OUTDIR_NAME = '01-classifier'

PATIENCE = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, help='dataset name')
    parser.add_argument('-e', '--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--full', action='store_true', help='If this flag is true, all train images are used for training')
    args = parser.parse_args()

    dataset_dir = os.path.join(DATASETS_DIR, args.dataset_name)
    outdir = os.path.join(WORKSPACE, args.dataset_name, OUTDIR_NAME)
    os.makedirs(outdir, exist_ok=True)

    # prepare_dataset
    x_train, x_test, y_train, y_test = prepare_dataset(dataset_dir, IMGLEN, args.full, 0.2)

    print('train shape:', x_train.shape, y_train.shape)
    print('test shape:', x_test.shape, y_test.shape)

    # learning
    model = CNN(IMGLEN)

    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0, mode='auto')

    history = model.fit(x_train, y_train,
                        batch_size=10,
                        epochs=args.epoch,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[early_stop])

    # save history
    histdf = pd.DataFrame(history.history)
    with open(os.path.join(outdir, 'history.csv'), 'w') as f:
        histdf.to_csv(f)

    # save model wights
    model.save_weights(os.path.join(outdir, 'weights.hdf5'))


def prepare_dataset(dataset_dir, imglen, is_full, test_size=0.2):
    # load
    train_a_dir = os.path.join(dataset_dir, TRAIN_A)
    train_b_dir = os.path.join(dataset_dir, TRAIN_B)
    train_a_images = load_images(train_a_dir, imglen)
    train_b_images = load_images(train_b_dir, imglen)
    
    if is_full:
        N_train = train_a_images.shape[0]
        
        x_train = np.append(train_a_images, train_b_images, axis=0)
        y_train = np.array([0]*N_train + [1]*N_train)
        
        test_a_dir = os.path.join(dataset_dir, TEST_A)
        test_b_dir = os.path.join(dataset_dir, TEST_B)
        test_a_images = load_images(test_a_dir, imglen)
        test_b_images = load_images(test_b_dir, imglen)
                
        x_test = np.append(test_a_images, test_b_images, axis=0)
        y_test = np.array([0]*test_a_images.shape[0] + [1]*test_b_images.shape[0])

    else:
        N_test = int(train_a_images.shape[0]*test_size)
        N_train = train_a_images.shape[0] - N_test

        x_train = np.append(train_a_images[:N_train], train_b_images[:N_train], axis=0)
        y_train = np.array([0]*N_train + [1]*N_train)
        x_test = np.append(train_a_images[N_train:], train_b_images[N_train:], axis=0)
        y_test = np.array([0]*N_test + [1]*N_test)

    # normalize
    x_train /= 255
    x_test /= 255

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    main()
