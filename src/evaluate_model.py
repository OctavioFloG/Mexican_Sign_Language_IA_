import tensorflow as tf
import numpy as np

def evaluate_model(model_path, X_test, y_test):
    model = tf.keras.models.load_model(model_path)
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Precisión en test: {acc*100:.2f}%")
    print(f"Pérdida en test: {loss:.4f}")
    return acc
