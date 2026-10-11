
import tkinter as tk
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

# Load the trained model
import os
import sys

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_path, "digit_model_improved.keras")
model = tf.keras.models.load_model(model_path)

# Create the main window FIRST
root = tk.Tk()
root.title("Handwritten Digit Recognizer")

# Create a 28 x 28 drawing grid, enlarged 10x on screen
grid_size = 28
pixel_size = 10
canvas_size = grid_size * pixel_size

canvas = tk.Canvas(
    root,
    width=canvas_size,
    height=canvas_size,
    bg="black",
    cursor="cross",
    highlightthickness=0
)
canvas.pack(padx=10, pady=10)

# Store the drawing at its actual 28 x 28 resolution
drawing = Image.new("L", (grid_size, grid_size), 0)
pen = ImageDraw.Draw(drawing)


def draw(event):
    # Convert screen coordinates to 28 x 28 pixel coordinates
    x = event.x // pixel_size
    y = event.y // pixel_size

    # Ignore coordinates outside the grid
    if not (0 <= x < grid_size and 0 <= y < grid_size):
        return

    # Fill the corresponding pixel on screen
    canvas.create_rectangle(
        x * pixel_size,
        y * pixel_size,
        (x + 1) * pixel_size,
        (y + 1) * pixel_size,
        fill="white",
        outline="white"
    )

    # Store the same pixel in the actual 28 x 28 image
    pen.point((x, y), fill=255)

canvas.bind("<B1-Motion>", draw)

def clear():
    canvas.delete("all")
    pen.rectangle((0, 0, grid_size - 1, grid_size - 1), fill=0)
    result_label.config(text="Draw a digit and click Predict")

def predict():
    # Resize the drawing to MNIST's 28 x 28 format
    pixels = np.array(drawing, dtype=np.float32) / 255.0
    
    pixels = pixels.reshape(1, 28, 28, 1)

    probabilities = model.predict(pixels, verbose=0)
    print(sorted(enumerate(probabilities[0]), key=lambda x: x[1], reverse=True)[:3])
    digit = int(np.argmax(probabilities[0]))
    confidence = float(np.max(probabilities[0])) * 100

    result_label.config(
        text=f"Predicted digit: {digit}\nConfidence: {confidence:.1f}%"
    )

tk.Button(root, text="Predict", command=predict).pack(pady=5)
tk.Button(root, text="Clear", command=clear).pack(pady=5)

result_label = tk.Label(
    root, text="Draw a digit and click Predict",
    font=("Arial", 14)
)
result_label.pack(pady=10)

root.mainloop()