# Introduction

The image below shows an example of an input image and an eye fixation or saliency map. The eye
fixation map is collected with the help of an eye tracker device. Many observers are shown the
image, and their eye movements are tracked. The eye fixation map is produced by averaging
over the fixations of many observers and smoothing with a Gaussian kernel. The objective of
this project is to implement a machine learning based system for eye fixation prediction. This
system is able to produce maps such as shown on the right given an input image.

<img src="example.png" width="50%" alt="Example of an input image (left) and the corresponding eye fixation map (right).">

# How to run

1. Install the required packages by running `pip install -r requirements.txt`.
2. Please name the data folder as `data` and place the data files in it (e.g. `data/train_fixations.txt`).
3. To train the model, run `python training.py`. The model will be saved after each epoch in the `models` folder.
4. To test the model, run `python testing.py`.

To view the live plot of the loss in TensorBoard, you can run `tensorboard --logdir=runs` in your command line and navigate to [localhost:6006](http://localhost:6006) in your web browser, then refresh the page (top right refresh button) as desired.

# Dataset

The dataset used in this project is split into three sets:

- Training, with 3006 images and eye fixation maps,
- Validation, with 1128 images and eye fixation maps, and
- Testing, with 1032 images.

Each image is of size 224-by-224, with 3 channels. Each fixation map is of size 224-by-224, with one channel. An example input image and fixation map is shown in the introduction section of this README.

Five text files `train_images.txt`, `train_fixations.txt`, `val_images.txt`, `val_fixations.txt`, and `test_images.txt` should also be provided.

Lines in `train_images.txt` and `train_fixations.txt` are paired: the ith row in both files points to a related pair of input image and eye fixation map. The same is true for `val_images.txt` and `val_fixations.txt`.