import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
from UI_app import AnimalApp
from SCNN_Model_Animals import SimpleCNN
import tkinter as tk

# Danh sách nhãn
categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

# === Xử lý đường dẫn khi đóng gói bằng PyInstaller ===
if getattr(sys, 'frozen', False):
    # Khi chạy dạng .exe
    base_path = sys._MEIPASS
else:
    # Khi chạy dạng .py thông thường
    base_path = os.path.dirname(__file__)

# Đường dẫn tới mô hình
model_path = os.path.join(base_path, "trained_models", "best_SCNN.pt")

# Load mô hình
model = SimpleCNN(num_class=10)
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model"])
model.eval()
softmax = nn.Softmax(dim=1)

# Hàm dự đoán
def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.transpose(image, (2, 0, 1)) / 255.0
    image = image[None, :, :, :]
    image = torch.from_numpy(image).float()

    with torch.no_grad():
        output = model(image)
        probs = softmax(output)
        max_idx = torch.argmax(probs)
        predicted_class = categories[max_idx]
        confidence = probs[0, max_idx].item()
        return predicted_class, confidence

# Khởi chạy ứng dụng
if __name__ == '__main__':
    root = tk.Tk()
    app = AnimalApp(root, predict_image)
    root.mainloop()