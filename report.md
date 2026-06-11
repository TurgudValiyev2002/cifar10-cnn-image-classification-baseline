# One-Page Report: CIFAR-10 CNN Baseline

## Motivation

We wanted a real computer vision baseline, not only linear models on pixels. CIFAR-10 is a useful test because object appearance varies strongly across images.

## Dataset

We used the official CIFAR-10 Python archive with a balanced subset of 8,000 training images and 2,000 test images across 10 classes.

## Method

We compared a most-frequent dummy classifier, an SGD linear classifier on flattened pixels, and a small PyTorch CNN. The CNN used three convolution blocks with batch normalization, ReLU, pooling, Adam optimization, learning rate 0.001, and 8 epochs.

## Results

The dummy classifier achieved 0.1000 accuracy. The raw-pixel linear model achieved 0.3480 accuracy and 0.3521 macro F1. The small CNN achieved 0.5255 accuracy and 0.5331 macro F1.

## Interpretation

The CNN is clearly stronger because it learns local spatial filters. The result is realistic and not inflated: 52.55% accuracy is a reasonable small-CNN baseline, not a final CIFAR-10 result.

## Conclusion

This project now matches its title. It gives a real CIFAR-10 CNN baseline and shows why convolution is useful for image classification.
