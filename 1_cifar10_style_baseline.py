from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RESULTS = Path("results")
RNG = np.random.default_rng(42)

def make_images(n=1500, size=16):
    images, labels = [], []
    for i in range(n):
        label = i % 5
        img = RNG.normal(0.15, 0.06, (size, size, 3))
        if label == 0:
            img[3:13, 3:13, 0] += 0.6
        elif label == 1:
            img[:, 6:10, 1] += 0.65
        elif label == 2:
            img[6:10, :, 2] += 0.65
        elif label == 3:
            np.fill_diagonal(img[:, :, 0], 0.9)
            np.fill_diagonal(np.fliplr(img[:, :, 1]), 0.8)
        else:
            rr, cc = np.ogrid[:size, :size]
            mask = (rr - size // 2) ** 2 + (cc - size // 2) ** 2 <= 16
            img[mask, :] += [0.45, 0.35, 0.15]
        images.append(np.clip(img, 0, 1))
        labels.append(label)
    return np.array(images), np.array(labels)

def conv_features(images):
    gray = images.mean(axis=3)
    feats = []
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = sobel_x.T
    for img, g in zip(images, gray):
        sx = np.zeros_like(g)
        sy = np.zeros_like(g)
        padded = np.pad(g, 1, mode="edge")
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                patch = padded[r:r+3, c:c+3]
                sx[r,c] = (patch * sobel_x).sum()
                sy[r,c] = (patch * sobel_y).sum()
        edge = np.sqrt(sx*sx + sy*sy)
        channel_means = img.mean(axis=(0,1))
        channel_stds = img.std(axis=(0,1))
        quadrants = [g[:8,:8].mean(), g[:8,8:].mean(), g[8:,:8].mean(), g[8:,8:].mean()]
        feats.append(np.r_[channel_means, channel_stds, edge.mean(), edge.max(), quadrants])
    return np.array(feats)

def main():
    RESULTS.mkdir(exist_ok=True)
    images, y = make_images()
    x_pixels = images.reshape(len(images), -1)
    x_conv = conv_features(images)
    xp_train, xp_test, xc_train, xc_test, y_train, y_test = train_test_split(x_pixels, x_conv, y, test_size=0.25, stratify=y, random_state=42)
    models = {
        "pixel_logistic_regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))]),
        "cnn_style_feature_logistic_regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))]),
    }
    rows = []
    for name, model in models.items():
        train_x, test_x = (xp_train, xp_test) if name.startswith("pixel") else (xc_train, xc_test)
        model.fit(train_x, y_train)
        pred = model.predict(test_x)
        rows.append({"model": name, "accuracy": round(accuracy_score(y_test, pred), 4), "macro_f1": round(f1_score(y_test, pred, average="macro"), 4)})
        pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).transpose().to_csv(RESULTS / f"{name}_classification_report.csv")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "model_metrics.csv", index=False)
    plt.figure(figsize=(6,4))
    plt.bar(metrics["model"], metrics["accuracy"], color=["#3d6fb6", "#4a8f5a"])
    plt.ylim(0,1.05)
    plt.ylabel("Accuracy")
    plt.title("CIFAR-Style Classification Baselines")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "accuracy_comparison.png", dpi=160)
    fig, axes = plt.subplots(1,5,figsize=(9,2))
    for cls, ax in enumerate(axes):
        ax.imshow(images[y == cls][0])
        ax.set_title(f"class {cls}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS / "sample_images.png", dpi=160)
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
