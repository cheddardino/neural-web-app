"""
Train DressCheck's HOG + SVM classifier and export deployment artifacts.
"""

from pathlib import Path
import pickle

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATASET_DIR = Path("Midterm Dataset")
MODEL_DIR = Path("models")
IMAGE_SIZE = (128, 128)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}

CLASSES = {
    "Non-Compliant": 0,
    "Compliant": 1,
}


def image_paths(folder):
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def extract_features(path):
    with Image.open(path) as img:
        arr = np.array(img.convert("RGB").resize(IMAGE_SIZE))

    return hog(
        arr,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=-1,
    )


def load_dataset():
    features = []
    labels = []
    skipped = []

    for class_name, label in CLASSES.items():
        folder = DATASET_DIR / class_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing dataset folder: {folder}")

        for path in image_paths(folder):
            try:
                features.append(extract_features(path))
                labels.append(label)
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                skipped.append((path, exc))

    if not features:
        raise RuntimeError("No valid training images were found.")

    return np.array(features), np.array(labels), skipped


def main():
    X, y, skipped = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    search = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid={
            "kernel": ["rbf", "linear"],
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto"],
            "class_weight": [None, "balanced"],
        },
        cv=5,
        n_jobs=1,
    )
    search.fit(X_train_scaled, y_train)
    model = search.best_estimator_

    accuracy = model.score(X_test_scaled, y_test)
    y_pred = model.predict(X_test_scaled)

    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)
    final_model = SVC(probability=True, random_state=42, **search.best_params_)
    final_model.fit(X_scaled, y)

    MODEL_DIR.mkdir(exist_ok=True)
    with (MODEL_DIR / "svm_model.pkl").open("wb") as f:
        pickle.dump(final_model, f)
    with (MODEL_DIR / "scaler.pkl").open("wb") as f:
        pickle.dump(final_scaler, f)

    print(f"Trained on {len(X_train)} images; tested on {len(X_test)} images.")
    print(f"Best params: {search.best_params_}")
    print(f"Accuracy: {accuracy:.3f}")
    print(classification_report(y_test, y_pred, target_names=["Non-Compliant", "Compliant"]))

    if skipped:
        print("Skipped unreadable files:")
        for path, exc in skipped:
            print(f"- {path}: {exc}")

    print("Saved models/svm_model.pkl")
    print("Saved models/scaler.pkl")


if __name__ == "__main__":
    main()
