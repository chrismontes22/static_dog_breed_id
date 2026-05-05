# test_classifier.py
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms

# ─── CONFIG ───────────────────────────────────────────────────────
ONNX_PATH = "dogmodel.onnx"
LABELS_PATH = "Dog_List.txt"

# ─── LOAD MODEL & LABELS ─────────────────────────────────────────
session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ─── PREPROCESSING (Must match your training exactly) ────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─── PREDICTION FUNCTION ─────────────────────────────────────────
def classify_image():
    filepath = filedialog.askopenfilename(
        title="Select a Dog Image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.webp")]
    )
    if not filepath:
        return

    try:
        # 1. Load & preprocess
        img = Image.open(filepath).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).numpy()  # Shape: [1, 3, 224, 224]

        # 2. Run ONNX inference
        outputs = session.run(None, {"input": input_tensor})[0]

        # 3. Softmax (numerically stable)
        logits = outputs[0]
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()

        # 4. Top-3
        top3_idx = np.argsort(probs)[-3:][::-1]
        results = "\n".join([f"{labels[i]:<30} {probs[i]*100:5.2f}%" for i in top3_idx])

        messagebox.showinfo("🐕 Top 3 Predictions", results)

    except Exception as e:
        messagebox.showerror("❌ Error", str(e))

# ─── TKINTER UI ──────────────────────────────────────────────────
root = tk.Tk()
root.title("Dog Breed Classifier (ONNX Test)")
root.geometry("320x180")
root.configure(bg="#111009")

tk.Label(root, text="Select an image to classify", 
         bg="#111009", fg="#E8A44A", font=("Segoe UI", 12)).pack(pady=(20, 10))

btn = tk.Button(root, text="📁 Choose Dog Photo", command=classify_image,
                font=("Segoe UI", 11, "bold"), bg="#E8A44A", fg="#1A1814",
                activebackground="#F0B660", relief="flat", padx=10, pady=5)
btn.pack(expand=True)

root.mainloop()