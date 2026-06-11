# One-Page Report: CIFAR-10 Image Classification Baselines

## Motivation

Real CIFAR-10 is a useful baseline dataset because objects have varied backgrounds, colors, poses, and textures. This makes the task much harder than simple rule-based image patterns.

## Dataset

We used the official CIFAR-10 Python dataset. The local experiment uses a balanced subset with 10,000 training images and 2,000 test images across 10 classes.

## Methods

We compared a most-frequent dummy classifier, an SGD linear classifier on flattened pixels, and an SGD linear classifier on compact visual features. The compact features include color statistics, quadrant brightness, and simple edge strength.

## Hyperparameters

The raw-pixel linear model used logistic loss, standardization, `max_iter=80`, and random seed 42. The compact-feature linear model used the same setup with `max_iter=120`.

## Results

The dummy classifier achieved 0.1000 accuracy and 0.0182 macro F1. The raw-pixel linear classifier achieved 0.3610 accuracy and 0.3636 macro F1. The compact-feature classifier achieved 0.3130 accuracy and 0.2860 macro F1.

## Interpretation

The result is realistic: simple linear models can learn some CIFAR-10 signal, but they are not strong enough for high-quality image classification.

## Conclusion

The project provides an honest real-data baseline. A small CNN should be the next model, and it should be compared against these baselines on the same subset.
