from __future__ import annotations

import copy
import pickle
import tarfile
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE = DATA / "cifar-10-python.tar.gz"
EXTRACTED = DATA / "cifar-10-batches-py"
SEED = 42
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def download_cifar10() -> None:
    DATA.mkdir(exist_ok=True)
    if not ARCHIVE.exists():
        print(f"Downloading CIFAR-10 from {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE)
    if not EXTRACTED.exists():
        with tarfile.open(ARCHIVE, "r:gz") as tar:
            tar.extractall(DATA, filter="data")


def load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    x = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(batch["labels"], dtype=np.int64)
    return x, y


def batch_path(name: str) -> Path:
    flat = DATA / name
    if flat.is_file():
        return flat
    standard = EXTRACTED / name
    if standard.is_file():
        return standard
    raise FileNotFoundError(f"Could not find CIFAR-10 batch file: {name}")


def load_cifar10_subset(train_per_class: int = 800, test_per_class: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    download_cifar10()
    train_images = []
    train_labels = []
    for batch_id in range(1, 6):
        x, y = load_batch(batch_path(f"data_batch_{batch_id}"))
        train_images.append(x)
        train_labels.append(y)
    x_train_all = np.concatenate(train_images)
    y_train_all = np.concatenate(train_labels)
    x_test_all, y_test_all = load_batch(batch_path("test_batch"))
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


class SmallCIFARCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


def make_cnn_tensors(images: np.ndarray, labels: np.ndarray) -> TensorDataset:
    x = images.astype(np.float32) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
    x = (x - mean) / std
    x = x.transpose(0, 3, 1, 2)
    return TensorDataset(torch.tensor(x), torch.tensor(labels, dtype=torch.long))


def evaluate_sklearn(name: str, model, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float | str]:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    pd.DataFrame(
        classification_report(y_test, pred, labels=list(range(10)), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    ).transpose().to_csv(RESULTS / f"{name}_classification_report.csv")
    pd.DataFrame(confusion_matrix(y_test, pred), index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(RESULTS / f"{name}_confusion_matrix.csv")
    return {"model": name, "accuracy": round(accuracy_score(y_test, pred), 4), "macro_f1": round(f1_score(y_test, pred, average="macro"), 4)}


def evaluate_cnn(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    labels = []
    preds = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            losses.append(loss_fn(logits, y_batch).item() * x_batch.size(0))
            labels.extend(y_batch.numpy())
            preds.extend(torch.argmax(logits, dim=1).numpy())
    return sum(losses) / len(loader.dataset), accuracy_score(labels, preds), np.array(labels), np.array(preds)


def train_cnn(x_train_img: np.ndarray, y_train: np.ndarray, x_test_img: np.ndarray, y_test: np.ndarray) -> tuple[dict[str, float | str], pd.DataFrame, np.ndarray, np.ndarray]:
    set_seed()
    rng = np.random.default_rng(SEED)
    train_idx = []
    val_idx = []
    for label in range(10):
        idx = np.where(y_train == label)[0]
        rng.shuffle(idx)
        split = int(0.85 * len(idx))
        train_idx.extend(idx[:split])
        val_idx.extend(idx[split:])
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    train_loader = DataLoader(make_cnn_tensors(x_train_img[train_idx], y_train[train_idx]), batch_size=128, shuffle=True)
    val_loader = DataLoader(make_cnn_tensors(x_train_img[val_idx], y_train[val_idx]), batch_size=256, shuffle=False)
    test_loader = DataLoader(make_cnn_tensors(x_test_img, y_test), batch_size=256, shuffle=False)

    model = SmallCIFARCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    history = []
    best_state = None
    best_val_acc = -1.0
    best_epoch = 0
    epochs = 8
    for epoch in range(1, epochs + 1):
        model.train()
        train_labels = []
        train_preds = []
        train_loss_total = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * x_batch.size(0)
            train_labels.extend(y_batch.numpy())
            train_preds.extend(torch.argmax(logits.detach(), dim=1).numpy())
        val_loss, val_acc, _, _ = evaluate_cnn(model, val_loader, loss_fn)
        train_acc = accuracy_score(train_labels, train_preds)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_total / len(train_loader.dataset),
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc, y_true, y_pred = evaluate_cnn(model, test_loader, loss_fn)
    pd.DataFrame(classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)).transpose().to_csv(
        RESULTS / "small_cnn_classification_report.csv"
    )
    pd.DataFrame(confusion_matrix(y_true, y_pred), index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(RESULTS / "small_cnn_confusion_matrix.csv")
    torch.save(model.state_dict(), RESULTS / "small_cifar_cnn.pt")
    row = {
        "model": "small_cnn",
        "accuracy": round(test_acc, 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "test_loss": round(test_loss, 4),
        "best_epoch": best_epoch,
    }
    return row, pd.DataFrame(history), y_true, y_pred


def plot_outputs(metrics: pd.DataFrame, history: pd.DataFrame, x_test: np.ndarray, y_test: np.ndarray) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(metrics["model"], metrics["accuracy"], color=["#888888", "#3d6fb6", "#2b8c5a"])
    plt.ylim(0, max(0.65, metrics["accuracy"].max() + 0.08))
    plt.ylabel("Accuracy")
    plt.title("CIFAR-10 Baselines and Small CNN")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "accuracy_comparison.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0].set_title("CNN loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="validation")
    axes[1].set_title("CNN accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "cnn_training_curves.png", dpi=180)
    plt.close(fig)

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
        ("Real CIFAR-10\n32x32 RGB", 0.13),
        ("Dummy + linear\nbaselines", 0.37),
        ("Small CNN\nConv-BN-ReLU", 0.62),
        ("Test metrics\nand errors", 0.86),
    ]
    for text, x in boxes:
        ax.text(x, 0.55, text, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef6ff", edgecolor="#336699"))
    for start, end in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(end[1] - 0.12, 0.55), xytext=(start[1] + 0.12, 0.55), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("Real CIFAR-10 CNN baseline workflow", fontsize=15)
    fig.tight_layout()
    fig.savefig(ASSETS / "readme_project_overview.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    x_train_img, y_train, x_test_img, y_test = load_cifar10_subset()
    x_train_pixels = pixel_features(x_train_img)
    x_test_pixels = pixel_features(x_test_img)
    rows = [
        evaluate_sklearn("most_frequent_dummy", DummyClassifier(strategy="most_frequent"), x_train_pixels, y_train, x_test_pixels, y_test),
        evaluate_sklearn(
            "raw_pixel_sgd_linear",
            Pipeline([("scaler", StandardScaler()), ("model", SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=80, random_state=SEED))]),
            x_train_pixels,
            y_train,
            x_test_pixels,
            y_test,
        ),
    ]
    cnn_row, history, _, _ = train_cnn(x_train_img, y_train, x_test_img, y_test)
    rows.append(cnn_row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "model_metrics.csv", index=False)
    history.to_csv(RESULTS / "cnn_training_history.csv", index=False)
    pd.DataFrame(
        [
            {"setting": "dataset", "value": "CIFAR-10 official Python archive"},
            {"setting": "train_images", "value": len(x_train_img)},
            {"setting": "test_images", "value": len(x_test_img)},
            {"setting": "classes", "value": 10},
            {"setting": "train_per_class", "value": 800},
            {"setting": "test_per_class", "value": 200},
            {"setting": "cnn_epochs", "value": 8},
            {"setting": "cnn_optimizer", "value": "Adam lr=0.001 weight_decay=0.0001"},
        ]
    ).to_csv(RESULTS / "experiment_setup.csv", index=False)
    plot_outputs(metrics, history, x_test_img, y_test)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
