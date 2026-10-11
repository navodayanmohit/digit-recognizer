
from pathlib import Path
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

# Keep the model in the same folder as this script
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "digit_model.keras"
IMPROVED_MODEL_PATH = BASE_DIR / "digit_model_improved.keras"



def create_model():
    """Create a small CNN for handwritten digit recognition."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(
            16, (3, 3), activation="relu"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model



def load_data():
    """Load MNIST and normalize its pixel values."""
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    return x_train, y_train, x_test, y_test

def load_emnist(split):
    dataset = tfds.load(
    "emnist/digits",
    split=split,
    as_supervised=True,
    shuffle_files=False
    )

    images, labels = [], []

    for image, label in tfds.as_numpy(dataset):
        image = np.transpose(image.squeeze(), (1, 0))
        images.append(image)
        labels.append(label)

    return (
        np.asarray(images, dtype=np.uint8),
        np.asarray(labels, dtype=np.int64)
    )

def train_model(continue_training=False):
    """Train a new model or continue training a saved model."""
    if continue_training:
        if not MODEL_PATH.exists():
            print("No saved model found. Choose Train from scratch first.")
            return

        model = tf.keras.models.load_model(MODEL_PATH)
        print("Loaded existing model. Continuing training...")

    else:
        model = create_model()
        print("Created a new model. Training from scratch...")

    x_train, y_train, _, _ = load_data()

    x_emnist, y_emnist = load_emnist("train")

    x_train = (x_train * 255).astype(np.uint8)

    x_train = np.concatenate([x_train, x_emnist], axis=0)
    y_train = np.concatenate([y_train, y_emnist], axis=0)

    x_train = x_train.astype(np.float32) / 255.0
    x_train = x_train[..., np.newaxis]

    del x_emnist, y_emnist

    try:
        epochs = int(input("Number of additional epochs (e.g. 5): "))
        if epochs < 1:
            print("Epochs must be at least 1.")
            return
    except ValueError:
        print("Please enter a valid whole number.")
        return

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1
    )

    model.save(MODEL_PATH.with_name("digit_model_improved.keras"))
print(f"\nModel saved to: {MODEL_PATH}")


def evaluate_model():
    """Evaluate the saved model on unseen test images."""
    if not MODEL_PATH.exists():
        print("No saved model found. Train a model first.")
        return

    _, _, x_test, y_test = load_data()
    model = tf.keras.models.load_model(MODEL_PATH.with_name("digit_model_improved.keras"))

    test_loss, test_accuracy = model.evaluate(
        x_test, y_test, verbose=0
    )

    print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")


def main():
    while True:
        print("\n===== HANDWRITTEN DIGIT RECOGNIZER =====")
        print("1. Train a new model from scratch")
        print("2. Continue training the saved model")
        print("3. Evaluate the saved model")
        print("4. Exit")

        choice = input("Choose an option (1-4): ").strip()

        if choice == "1":
            train_model(continue_training=False)
        elif choice == "2":
            train_model(continue_training=True)
        elif choice == "3":
            evaluate_model()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please choose 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()