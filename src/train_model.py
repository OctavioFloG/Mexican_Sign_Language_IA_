import tensorflow as tf
import matplotlib.pyplot as plt
import os

def train_cnn(X_train, y_train, X_valid, y_valid, num_classes, save_path="models/modelo_lsm_cnn.h5"):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(64,64,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid), batch_size=16)

    loss, acc = model.evaluate(X_valid, y_valid)
    print(f"Precisión en validación: {acc*100:.2f}%")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Modelo guardado en {save_path}")

    plt.plot(history.history["accuracy"], label="Entrenamiento")
    plt.plot(history.history["val_accuracy"], label="Validación")
    plt.title("Precisión del modelo")
    plt.xlabel("Épocas")
    plt.ylabel("Precisión")
    plt.legend()
    plt.show()
