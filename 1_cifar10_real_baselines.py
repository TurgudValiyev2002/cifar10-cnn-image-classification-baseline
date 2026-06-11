from __future__ import annotations

import pickle
import tarfile
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE = DATA / "cifar-10-python.tar.gz"
EXTRACTED = DATA / "cifar-10-batches-py"
SEED = 42
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def download_cifar10() -> None:
    DATA.mkdir(exist_ok=True)
    if not ARCHIVE.exists():
        print(f"Downloading CIFAR-10 from {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE)
    if not EXTRACTED.exists():
        with tarfile.open(ARCHIVE, "r:gz") as tar:
            tar.extractall(DATA)


def load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    x = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(batch["labels"], dtype=np.int64)
    return x, y


def load_cifar10_subset(train_per_class: int = 1000, test_per_class: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    download_cifar10()
    train_images = []
    train_labels = []
    for batch_id in range(1, 6):
        x, y = load_batch(EXTRACTED / f"data_batch_{batch_id}")
        train_images.append(x)
        train_labels.append(y)
    x_train_all = np.concatenate(train_images)
    y_train_all = np.concatenate(train_labels)
    x_test_all, y_test_all = load_batch(EXTRACTED / "test_batch")

    rng = np.random.default_rng(SEED)

    def balanced_subset(x: np.ndarray, y: np.ndarray, per_class: int) -> tuple[np.ndarray, np.ndarray]:
        indices = []
        for label in range(10):
            label_idx = np.where(y == label)[0]
            indices.extend(rng.choice(label_idx, size=per_class, replace=False))
        indices = np.array(indices)
        rng.shuffle(indices)
        return x[indices], y[indices]

    return (*balanced_subset(x_train_all, y_train_all, train_per_class), *balanced_subset(x_test_all, y_test_all, test_per_class))


def pixel_features(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32).reshape(len(images), -1) / 255.0


def compact_visual_features(images: np.ndarray) -> np.ndarray:
    images = images.astype(np.float32) / 255.0
    gray = images.mean(axis=3)
    channel_mean = images.mean(axis=(1, 2))
    channel_std = images.std(axis=(1, 2))
    quadrant_means = []
    for r0, r1, c0, c1 in [(0, 16, 0, 16), (0, 16, 16, 32), (16, 32, 0, 16), (16, 32, 16, 32)]:
        quadrant_means.append(gray[:, r0:r1, c0:c1].mean(axis=(1, 2)))
    vertical_edges = np.abs(gray[:, :, 1:] - gray[:, :, :-1]).mean(axis=(1, 2))
    horizontal_edges = np.abs(gray[:, 1:, :] - gray[:, :-1, :]).mean(axis=(1, 2))
    return np.column_stack([channel_mean, channel_std, *quadrant_means, vertical_edges, horizontal_edges])


def evaluate_model(name: str, model, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float | str]:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    report = classification_report(
        y_test,
        pred,
        labels=list(range(10)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(RESULTS / f"{name}_classification_report.csv")
    pd.DataFrame(confusion_matrix(y_test, pred), index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        RESULTS / f"{name}_confusion_matrix.csv"
    )
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
    }


def plot_outputs(metrics: pd.DataFrame, x_test: np.ndarray, y_test: np.ndarray) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(metrics["model"], metrics["accuracy"], color=["#888888", "#3d6fb6", "#4a8f5a"])
    plt.ylim(0, 0.45)
    plt.ylabel("Accuracy")
    plt.title("Real CIFAR-10 Baseline Accuracy")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "accuracy_comparison.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for label, ax in enumerate(axes.ravel()):
        image = x_test[y_test == label][0]
        ax.imshow(image)
        ax.set_title(CLASS_NAMES[label], fontsize=9)
        ax.axis("off")
    fig.suptitle("Real CIFAR-10 examples", y=0.98)
    fig.tight_layout()
    fig.savefig(RESULTS / "sample_images.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    boxes = [
        ("Real CIFAR-10\n32x32 RGB images", 0.13),
        ("Pixel or compact\nvisual features", 0.38),
        ("Linear baselines\nnot a CNN", 0.62),
        ("Test metrics +\nerror analysis", 0.86),
    ]
    for text, x in boxes:
        ax.text(
            x,
            0.55,
            text,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef6ff", edgecolor="#336699"),
        )
    for start, end in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(end[1] - 0.12, 0.55), xytext=(start[1] + 0.12, 0.55), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("Real CIFAR-10 classification baseline workflow", fontsize=15)
    fig.tight_layout()
    fig.savefig(ASSETS / "readme_project_overview.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    x_train_img, y_train, x_test_img, y_test = load_cifar10_subset()
    x_train_pixels = pixel_features(x_train_img)
    x_test_pixels = pixel_features(x_test_img)
    x_train_compact = compact_visual_features(x_train_img)
    x_test_compact = compact_visual_features(x_test_img)

    rows = [
        evaluate_model("most_frequent_dummy", DummyClassifier(strategy="most_frequent"), x_train_pixels, y_train, x_test_pixels, y_test),
        evaluate_model(
            "raw_pixel_sgd_linear",
            Pipeline([("scaler", StandardScaler()), ("model", SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=80, random_state=SEED))]),
            x_train_pixels,
            y_train,
            x_test_pixels,
            y_test,
        ),
        evaluate_model(
            "compact_feature_sgd_linear",
            Pipeline([("scaler", StandardScaler()), ("model", SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=120, random_state=SEED))]),
            x_train_compact,
            y_train,
            x_test_compact,
            y_test,
        ),
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "model_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"setting": "dataset", "value": "CIFAR-10 official Python archive"},
            {"setting": "train_images", "value": len(x_train_img)},
            {"setting": "test_images", "value": len(x_test_img)},
            {"setting": "classes", "value": 10},
            {"setting": "train_per_class", "value": 1000},
            {"setting": "test_per_class", "value": 200},
        ]
    ).to_csv(RESULTS / "experiment_setup.csv", index=False)
    plot_outputs(metrics, x_test_img, y_test)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
