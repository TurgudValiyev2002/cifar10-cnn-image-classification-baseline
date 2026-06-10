# Report: CIFAR-10-Style Image Classification Baseline

## Motivation

We built an image-classification baseline to compare raw-pixel features with convolution-style features.

## Dataset

The dataset contains 1500 controlled 16x16 RGB images with 5 visual classes. It is not the real CIFAR-10 dataset.

## Method

We trained logistic regression on raw pixels and on hand-built convolution-style features.

## Hyperparameters

The test split was 25 percent. Logistic regression used `max_iter=1000` and `random_state=42`.

## Results

Both models achieved 1.0000 accuracy and 1.0000 macro F1.

## Interpretation

The task is too simple for a serious model comparison. The result confirms that the pipeline works, but not that the model is strong on real CIFAR-10.

## Conclusion

This is a clean baseline workflow. A future version should use real CIFAR-10 and a CNN.
