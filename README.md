# CIFAR-10 Image Classification Baselines

![Project overview](assets/readme_project_overview.png)

Figure: real CIFAR-10 images are converted into pixel or compact visual features, then evaluated with simple linear baselines.

## Motivation

CIFAR-10 is a standard computer vision benchmark. It is much harder than a controlled synthetic image dataset because object pose, background, color, and texture vary strongly. A simple baseline is useful because it tells us what performance we get before using a CNN.

## Project Goal

We evaluated simple classification baselines on the real CIFAR-10 dataset. The goal was to replace the earlier too-easy synthetic setup with a real image dataset and report honest, non-perfect results.

## Dataset

We used the official CIFAR-10 Python archive from the University of Toronto. CIFAR-10 has 10 object classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

For a fast local experiment, the script uses a balanced subset:

- Training images: 10,000
- Test images: 2,000
- Classes: 10
- Train images per class: 1,000
- Test images per class: 200
- Image size: 32x32 RGB

The dataset is downloaded by the script into `data/`, which is ignored by Git.

## Tools

Python, NumPy, pandas, scikit-learn, and matplotlib.

## Methods

We compared three baselines:

- Most-frequent dummy classifier
- SGD linear classifier on flattened RGB pixels
- SGD linear classifier on compact visual features such as color statistics, quadrant intensity, and simple edge strength

These are not CNNs. They are intentionally simple baselines.

## Hyperparameters

| Setting | Value |
|---|---:|
| Train subset | 10,000 images |
| Test subset | 2,000 images |
| Optimizer/model | SGD linear classifier |
| Loss | Logistic loss |
| Raw-pixel max iterations | 80 |
| Compact-feature max iterations | 120 |
| Random seed | 42 |

## Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Most-frequent dummy | 0.1000 | 0.0182 |
| Raw pixel SGD linear | 0.3610 | 0.3636 |
| Compact feature SGD linear | 0.3130 | 0.2860 |

![Accuracy comparison](results/accuracy_comparison.png)

![Sample CIFAR-10 images](results/sample_images.png)

Result files include:

- `results/model_metrics.csv`
- `results/experiment_setup.csv`
- `results/*_classification_report.csv`
- `results/*_confusion_matrix.csv`

## Interpretation

The real CIFAR-10 result is no longer perfect, which is good. The raw-pixel linear model reached 36.10% accuracy, clearly above the 10% dummy baseline but far below what a CNN can achieve. This means flattened pixels contain useful signal, but linear decision boundaries are too weak for full object recognition.

The compact feature model was also above the dummy baseline but weaker than raw pixels. Hand-built color and edge summaries lose too much spatial detail for CIFAR-10.

## Conclusion

This project now gives a realistic CIFAR-10 baseline. The correct next step is to train a small CNN and compare it against these linear baselines using the same train/test subset.

## How To Run

```bash
pip install -r requirements.txt
python 1_cifar10_real_baselines.py
```
