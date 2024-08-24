# CNN classification

Classify images with CNN.

## Requirement

- python 3.9.19

## Install

`pip install -r requirements.txt`

## Usage

### 0. Prepare dataset

Make dataset directory. In this case, let's prepare a dataset named "mydata".

`mkdir -p datasets/mydata/{trainA,trainB,testA,testB}`

And place your images in each directories.

Each image sohuld be trimmed to square.

### 1. Training

`./01-training.py mydata`

### 2. Test

`./02-test.py mydata`

### 3. GradCam

`./03-grad-cam.py mydata`

### 4. Draw heatmap

`./04-draw-heatmaps.py mydata`

## Author

[ShimeiYago](https://github.com/ShimeiYago)
