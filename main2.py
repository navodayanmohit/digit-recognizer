from pathlib import Path
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

# Model paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "digit_model.keras"
IMPROVED_MODEL_PATH = BASE_DIR / "digit_model_improved.keras"


def create_model():
    """Create a small CNN for handwritten digit recognition."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
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
    """Load and normalize MNIST."""
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return x_train, y_train, x_test, y_test


def load_emnist(split):
    """Load EMNIST Digits and transpose its images."""
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

    images = np.asarray(images, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.int64)

    images = images.astype(np.float32) / 255.0
    images = images[..., np.newaxis]

    return images, labels


def evaluate_results(model, x_mnist, y_mnist, x_emnist, y_emnist, title):
    """Evaluate the model on both datasets."""
    mnist_loss, mnist_accuracy = model.evaluate(
        x_mnist, y_mnist, verbose=0
    )

    emnist_loss, emnist_accuracy = model.evaluate(
        x_emnist, y_emnist, verbose=0
    )

    print(f"\n===== {title} =====")
    print(f"MNIST Test Accuracy:  {mnist_accuracy * 100:.2f}%")
    print(f"MNIST Test Loss:      {mnist_loss:.4f}")
    print(f"EMNIST Test Accuracy: {emnist_accuracy * 100:.2f}%")
    print(f"EMNIST Test Loss:     {emnist_loss:.4f}")


def train_model(continue_training=False):
    """Train on MNIST, test, then continue training on EMNIST."""
    if continue_training:
        if not IMPROVED_MODEL_PATH.exists():
            print("No improved model found. Train from scratch first.")
            return

        model = tf.keras.models.load_model(IMPROVED_MODEL_PATH)
        print("Loaded improved model. Continuing training on EMNIST.")

    else:
        model = create_model()
        print("Created a new model. Training on MNIST first.")

    try:
        epochs = int(input("Number of epochs for each dataset (e.g. 5): "))
        if epochs < 1:
            print("Epochs must be at least 1.")
            return
    except ValueError:
        print("Please enter a valid whole number.")
        return

    # Load test data for both datasets.
    x_train, y_train, x_mnist_test, y_mnist_test = load_data()
    x_emnist_test, y_emnist_test = load_emnist("test")

    if not continue_training:
        # Stage 1: Train on MNIST.
        model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1
        )

        # Save to a separate file; preserve the original model.
        model.save(IMPROVED_MODEL_PATH)

        # Test after MNIST training.
        evaluate_results(
            model,
            x_mnist_test,
            y_mnist_test,
            x_emnist_test,
            y_emnist_test,
            "AFTER MNIST TRAINING"
        )

    # Stage 2: Continue training the same model on EMNIST.
    print("\nLoading EMNIST training data...")
    x_emnist_train, y_emnist_train = load_emnist("train")

    print("\nContinuing training on EMNIST...")
    model.fit(
        x_emnist_train,
        y_emnist_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1
    )

    # Save the updated model separately.
    model.save(IMPROVED_MODEL_PATH)

    # Test after EMNIST training on both datasets.
    evaluate_results(
        model,
        x_mnist_test,
        y_mnist_test,
        x_emnist_test,
        y_emnist_test,
        "AFTER EMNIST TRAINING"
    )

    print(f"\nImproved model saved to: {IMPROVED_MODEL_PATH}")


def evaluate_model():
    """Evaluate the improved model on MNIST and EMNIST test data."""
    if not IMPROVED_MODEL_PATH.exists():
        print("No improved model found. Train a model first.")
        return

    model = tf.keras.models.load_model(IMPROVED_MODEL_PATH)

    _, _, x_mnist_test, y_mnist_test = load_data()
    x_emnist_test, y_emnist_test = load_emnist("test")

    evaluate_results(
        model,
        x_mnist_test,
        y_mnist_test,
        x_emnist_test,
        y_emnist_test,
        "IMPROVED MODEL TEST RESULTS"
    )


def main():
    while True:
        print("\n===== HANDWRITTEN DIGIT RECOGNIZER =====")
        print("1. Train a new model from scratch")
        print("2. Continue training the improved model on EMNIST")
        print("3. Evaluate the improved model")
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