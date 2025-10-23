import os
import json
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_coco_split(split_dir, img_size=(64, 64)):
    """
    Carga un split del dataset (train o valid) desde formato COCO.
    """
    json_path = os.path.join(split_dir, "_annotations.coco.json")
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    X, y = [], []
    img_id_to_path = {img["id"]: os.path.join(split_dir, img["file_name"]) for img in coco["images"]}

    for ann in coco["annotations"]:
        img_path = img_id_to_path.get(ann["image_id"])
        if img_path and os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(ann["category_id"])

    X = np.array(X, dtype="float32") / 255.0
    y = to_categorical([list(categories.keys()).index(lbl) for lbl in y], num_classes=len(categories))
    return X, y, categories


def load_sign_language_dataset(base_dir="dataset/sign_language", img_size=(64, 64)):
    """
    Carga el dataset completo (train y valid) desde la estructura de carpetas.
    """
    train_dir = os.path.join(base_dir, "train")
    valid_dir = os.path.join(base_dir, "valid")

    X_train, y_train, categories = load_coco_split(train_dir, img_size)
    X_valid, y_valid, _ = load_coco_split(valid_dir, img_size)

    print(f"Train: {len(X_train)} imágenes | Valid: {len(X_valid)} imágenes | Clases: {len(categories)}")
    return X_train, y_train, X_valid, y_valid, categories
