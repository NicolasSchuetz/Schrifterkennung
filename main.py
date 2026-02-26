import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading


# MODEL
model = tf.keras.models.load_model("model.keras")
classes = [chr(i) for i in range(65, 91)]

CANVAS_SIZE = 320
MODEL_SIZE = 32

# =========================
# WINDOW
# =========================
root = tk.Tk()
root.title("🧠 AI Buchstaben-Erkennung")
root.geometry("900x400")
root.configure(bg="#1e1e1e")

main_frame = tk.Frame(root, bg="#1e1e1e")
main_frame.pack(fill="both", expand=True)

# =========================
# LEFT: CANVAS
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
# RIGHT: PLOT
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
# DRAW SMOOTH LINE
# =========================
last_x, last_y = None, None

def paint(event):
    global last_x, last_y
    
    if last_x is not None:
        canvas.create_line(last_x, last_y, event.x, event.y,
                           width=15, fill="black",
                           capstyle=tk.ROUND, smooth=True)
        draw.line([last_x, last_y, event.x, event.y],
                  fill="black", width=15)

    last_x, last_y = event.x, event.y

def reset_line(event):
    global last_x, last_y
    last_x, last_y = None, None

canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", reset_line)


# THREAD SAFE PREDICTION
prediction_running = False

def run_prediction():
    global prediction_running
    prediction_running = True

    img = image.resize((MODEL_SIZE, MODEL_SIZE))
    img = ImageOps.invert(img)

    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, MODEL_SIZE, MODEL_SIZE, 1)

    prediction = model.predict(img_array, verbose=0)[0]

    # UI Update im Main Thread
    root.after(0, update_plot, prediction)

    prediction_running = False

def update_plot(prediction):
    for bar, prob in zip(bars, prediction):
        bar.set_height(prob)
    canvas_plot.draw()

def prediction_loop():
    if not prediction_running:
        threading.Thread(target=run_prediction).start()
    root.after(120, prediction_loop)  # alle 120ms

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