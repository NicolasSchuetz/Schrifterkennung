import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# =========================
# MODEL LADEN
# =========================
model = tf.keras.models.load_model("model.keras")
classes = [chr(i) for i in range(65, 91)]  # A-Z

CANVAS_SIZE = 320
MODEL_SIZE = 32

# =========================
# WINDOW SETUP
# =========================
root = tk.Tk()
root.title("🧠 AI Buchstaben-Erkennung")
root.geometry("950x420")
root.configure(bg="#1e1e1e")

main_frame = tk.Frame(root, bg="#1e1e1e")
main_frame.pack(fill="both", expand=True)

# =========================
# ZEICHENBEREICH
# =========================
left_frame = tk.Frame(main_frame, bg="#1e1e1e")
left_frame.pack(side="left", padx=20, pady=20)

canvas = tk.Canvas(
    left_frame,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="white",
    highlightthickness=0
)
canvas.pack()

image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "white")
draw = ImageDraw.Draw(image)

# =========================
# DIAGRAMM
# =========================
right_frame = tk.Frame(main_frame, bg="#1e1e1e")
right_frame.pack(side="right", padx=20, pady=20)

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(classes, np.zeros(len(classes)))
ax.set_ylim(0, 1)
ax.set_facecolor("#2b2b2b")
fig.patch.set_facecolor("#1e1e1e")
ax.tick_params(axis='x', colors='white', rotation=90)
ax.tick_params(axis='y', colors='white')

canvas_plot = FigureCanvasTkAgg(fig, master=right_frame)
canvas_plot.get_tk_widget().pack()

# =========================
# SMOOTH DRAWING
# =========================
last_x, last_y = None, None

def paint(event):
    global last_x, last_y

    if last_x is not None:
        canvas.create_line(
            last_x, last_y, event.x, event.y,
            width=15, fill="black",
            capstyle=tk.ROUND, smooth=True
        )
        draw.line(
            [last_x, last_y, event.x, event.y],
            fill="black", width=15
        )

    last_x, last_y = event.x, event.y

def reset_line(event):
    global last_x, last_y
    last_x, last_y = None, None

canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", reset_line)

# =========================
# PREPROCESSING (Crop + Center)
# =========================
def preprocess_image():
    img_array = np.array(image)

    # Invertieren
    img_array = 255 - img_array

    # Rauschen entfernen
    img_array[img_array < 20] = 0

    if np.sum(img_array) == 0:
        return None

    coords = np.column_stack(np.where(img_array > 0))
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = img_array[y_min:y_max, x_min:x_max]

    h, w = cropped.shape
    size = max(h, w)

    square = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

    img_resized = Image.fromarray(square).resize((MODEL_SIZE, MODEL_SIZE))

    img_final = np.array(img_resized).astype("float32") / 255.0
    img_final = img_final.reshape(1, MODEL_SIZE, MODEL_SIZE, 1)

    return img_final

# =========================
# THREAD SAFE PREDICTION
# =========================
prediction_running = False

def run_prediction():
    global prediction_running
    prediction_running = True

    img_array = preprocess_image()

    if img_array is not None:
        prediction = model.predict(img_array, verbose=0)[0]
        root.after(0, update_plot, prediction)

    prediction_running = False

def update_plot(prediction):
    for bar, prob in zip(bars, prediction):
        bar.set_height(prob)
    canvas_plot.draw()

def prediction_loop():
    if not prediction_running:
        threading.Thread(target=run_prediction).start()
    root.after(120, prediction_loop)

prediction_loop()

# =========================
# CLEAR
# =========================
def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill="white")

    for bar in bars:
        bar.set_height(0)

    canvas_plot.draw()

clear_btn = ttk.Button(left_frame, text="Clear", command=clear)
clear_btn.pack(pady=10)

root.mainloop()