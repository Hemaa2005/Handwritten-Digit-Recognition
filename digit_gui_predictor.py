# digit_gui_predictor.py
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Load pre-trained CNN model
model = tf.keras.models.load_model("mnist_cnn_model_v2.h5")

# Initialize GUI
window = tk.Tk()
window.title("Digit Predictor (MNIST)")

canvas_width = 280
canvas_height = 280
white = (255, 255, 255)

# Setup canvas and PIL image
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

image1 = Image.new("RGB", (canvas_width, canvas_height), white)
draw = ImageDraw.Draw(image1)

# Draw on canvas and image
def draw_lines(event):
    x, y = event.x, event.y
    r = 8
    canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
    draw.ellipse([x - r, y - r, x + r, y + r], fill='black')

canvas.bind("<B1-Motion>", draw_lines)

# Prediction logic
def predict_digit():
    # Convert to grayscale and resize
    img = image1.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img = np.array(img).astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    # Predict digit
    pred = model.predict(img)
    digit = np.argmax(pred)
    confidence = np.max(pred)

    result_label.config(text=f"Predicted: {digit} (Confidence: {confidence:.2f})")

# Clear canvas and image
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill=white)
    result_label.config(text="Draw a digit!")

# Buttons
button_frame = tk.Frame(window)
button_frame.pack()
predict_button = tk.Button(button_frame, text="Predict", command=predict_digit)
predict_button.grid(row=0, column=0, padx=10, pady=10)

clear_button = tk.Button(button_frame, text="Clear", command=clear_canvas)
clear_button.grid(row=0, column=1, padx=10, pady=10)

result_label = tk.Label(window, text="Draw a digit!", font=("Helvetica", 16))
result_label.pack()

window.mainloop()
