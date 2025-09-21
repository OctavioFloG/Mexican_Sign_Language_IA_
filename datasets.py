import matplotlib.pyplot as plt
import cv2
import os

# Carpeta del dataset (ejemplo)
dataset_path = "dataset/sign_language/train/"

# Cargar algunas imágenes
images = [cv2.imread(os.path.join(dataset_path, f)) for f in os.listdir(dataset_path)[:5]]

# Mostrar ejemplos
plt.figure(figsize=(10,5))
for i, img in enumerate(images):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 5, i+1)
    plt.imshow(img_rgb)
    plt.axis("off")
plt.show()
