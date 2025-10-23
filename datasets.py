import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
dataset_dir = "dataset"  # carpeta donde descomprimiste el dataset
annotations_file = os.path.join(dataset_dir, "annotations.json")  # ajusta el nombre si cambia
img_size = (64, 64)  # tamaño de redimensionado

test_size = 0.2      # 80% train / 20% test

# --- LEER ANOTACIONES COCO ---
with open(annotations_file, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
num_classes = len(categories)

# --- CARGAR IMÁGENES Y ETIQUETAS ---
X, y = [], []
img_id_to_path = {img["id"]: os.path.join(dataset_dir, img["file_name"]) for img in coco_data["images"]}

for ann in coco_data["annotations"]:
    img_path = img_id_to_path[ann["image_id"]]
    if os.path.exists(img_path):
        # Cargar imagen
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)

        X.append(img)
        y.append(ann["category_id"])

X = np.array(X, dtype="float32") / 255.0
y = to_categorical([list(categories.keys()).index(lbl) for lbl in y], num_classes=num_classes)

print(f"Dataset cargado: {X.shape[0]} ejemplos, {num_classes} clases.")

# --- DIVIDIR EN TRAIN/TEST ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# --- DEFINIR MODELO CNN ---
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(img_size[0], img_size[1], 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# --- ENTRENAR ---
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=16)

# --- EVALUAR ---
loss, acc = model.evaluate(X_test, y_test)
print(f"Precisión en test: {acc*100:.2f}%")

# --- GRAFICAR CURVAS DE APRENDIZAJE ---
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validación")
plt.title("Precisión del modelo")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.legend()
plt.show()

# --- GUARDAR MODELO ---
model.save("modelo_lsm_cnn.h5")
print("Modelo guardado como modelo_lsm_cnn.h5")
