from src.dataset_loader import load_sign_language_dataset
from src.train_model import train_cnn
from src.visualize_samples import visualize_samples

# Cargar dataset COCO dividido
X_train, y_train, X_valid, y_valid, categories = load_sign_language_dataset("dataset/sign_language")

# Mostrar ejemplos
visualize_samples(X_train, y_train, categories)

# Entrenar modelo CNN
train_cnn(X_train, y_train, X_valid, y_valid, num_classes=len(categories))
