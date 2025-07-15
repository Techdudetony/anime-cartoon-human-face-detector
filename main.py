import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from keras.models import Sequential
from keras.initializers import HeNormal
from keras.utils import image_dataset_from_directory
from pathlib import Path

# Globals
IMG_SIZE = (64, 64)
BATCH = 32
EPOCHS = 30
NUM_CLASSES = 3 # anime, human, cartoon
DATASET_DIR = Path("dataset")
SEED = 42

# Set seed for reproducibility
tf.random.set_seed(SEED)

# Set initializer
initializer = HeNormal()

# Get Dataset Utility
def ds_load(path: Path):
    '''
    Loads images from directory.
    
    Returns:
        Preprocessed training and validation datasets
    '''
    # Train DS
    train_ds = image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH,
    )
    
    # Validation DS
    val_ds = image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH,
    )
    
    # Prefetch for speed
    autofetch = tf.data.AUTOTUNE
    return (
        train_ds.cache().prefetch(buffer_size=autofetch),
        val_ds.cache().prefetch(buffer_size=autofetch),
    )
    
# Build Model
def build_model():
    '''
    Builds and compiles the CNN model
    
    Returns:
        Compiled CNN model
    '''
    resize_and_rescale = keras.Sequential([
        layers.Resizing(*IMG_SIZE),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ])
    
    model = Sequential([
        resize_and_rescale,
        layers.Conv2D(32, 3, activation="relu", kernel_initializer=initializer),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, activation="relu", kernel_initializer=initializer),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu", kernel_initializer=initializer),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation="relu", kernel_initializer=initializer),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu", kernel_initializer=initializer),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(256, activation="relu", kernel_initializer=initializer),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# Train & Evaluate Model
def train():
    '''
    Trains the CNN model and plots performance
    '''
    train_ds, val_ds = ds_load(DATASET_DIR)
    train_count = train_ds.cardinality().numpy() * BATCH
    val_count = val_ds.cardinality().numpy() * BATCH
    model = build_model()
    
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
    )
    
    # Save model
    model.save("model/face_classifier.keras")
    
    # Plot training curves
    plot_history(history)
    
    # Save results
    save_results(history, train_count, val_count)
    
    # Visualize Image Predictions
    visualize_predictions(model)
    
def plot_history(history):
    # Loss plot
    plt.subplot(211)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.legend()
    
    # Accuracy Plot
    plt.subplot(212)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training/training_results3.png")
    plt.show()
    
def save_results(history, train_count, val_count, filename="training_summary.txt"):
    '''
    Saves training results to a .txt file.
    '''
    with open(filename, "w") as f:
        f.write("📊 CNN Training Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch Size: {BATCH}\n")
        f.write(f"Image Size: {IMG_SIZE}\n")
        f.write(f"Classes: {NUM_CLASSES}\n\n")
        
        f.write(f"Training Samples: {train_count}\n")
        f.write(f"Validation Samples: {val_count}\n\n")

        f.write("Final Training Accuracy: {:.4f}\n".format(history.history["accuracy"][-1]))
        f.write("Final Validation Accuracy: {:.4f}\n".format(history.history["val_accuracy"][-1]))
        f.write("Final Training Loss: {:.4f}\n".format(history.history["loss"][-1]))
        f.write("Final Validation Loss: {:.4f}\n".format(history.history["val_loss"][-1]))

def visualize_predictions(model, dataset_dir=DATASET_DIR, filename="inference/classification_grid.png"):
    '''
    Generates and saves a 3x3 image grid with predicted vs actual labels.
    '''
    class_names = sorted([item.name for item in dataset_dir.iterdir() if item.is_dir()])
    
    # Load a small shuffled dataset
    test_ds = image_dataset_from_directory(
        dataset_dir,
        image_size=IMG_SIZE,
        batch_size=1,
        shuffle=True,
        seed=SEED
    )
    
    # Collect 9 random samples
    images, labels = [], []
    for batch in test_ds.take(50):  # take up to 50 batches to gather 9 unique samples
        for i in range(len(batch[0])):
            images.append(batch[0][i])
            labels.append(batch[1][i].numpy())
            if len(images) == 9:
                break
        if len(images) == 9:
            break
    
    # Make predictions
    predictions = model.predict(tf.stack(images))
    pred_labels = tf.argmax(predictions, axis=1).numpy()
    
    # Plot
    os.makedirs("inference", exist_ok=True)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        true_lbl = class_names[labels[i]]
        pred_lbl = class_names[pred_labels[i]]
        color = "green" if true_lbl == pred_lbl else "red"
        plt.title(f"Pred: {pred_lbl}\nTrue: {true_lbl}", color=color, fontsize=9)
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# ----------------------------------    
#           MAIN
# ----------------------------------
if __name__ == "__main__":
    if not DATASET_DIR.exists():
        raise FileNotFoundError("Dataset not found. Run utils/download_sample.py first.")
    
    os.makedirs("model", exist_ok=True)
    train()