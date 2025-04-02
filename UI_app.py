# ui_app.py
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import sys

class AnimalApp:
    def __init__(self, root, predictor_function):
        self.root = root
        self.root.title("🐾 Animal Recognition App")
        self.root.geometry("800x500")
        self.root.configure(bg="white")

        self.predictor = predictor_function

        # === Xử lý đường dẫn cho avatar (tương thích PyInstaller) ===
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(__file__)
        avatar_path = os.path.join(base_path, "avatar", "avatar.jpg")

        # === Chia 2 khung trái - phải ===
        self.left_frame = tk.Frame(self.root, bg="white", width=300)
        self.left_frame.pack(side="left", fill="both")

        self.right_frame = tk.Frame(self.root, bg="white")
        self.right_frame.pack(side="right", fill="both", expand=True)

        # === Bên trái: Thông tin cá nhân ===
        tk.Label(
            self.left_frame,
            text="梁玉辉",
            font=("Arial", 18, "bold"),
            bg="white"
        ).pack(pady=(30, 10))

        # Ảnh avatar
        try:
            avatar = Image.open(avatar_path).resize((150, 150))
            avatar_tk = ImageTk.PhotoImage(avatar)
            self.avatar_label = tk.Label(self.left_frame, image=avatar_tk, bg="white")
            self.avatar_label.image = avatar_tk
            self.avatar_label.pack()
        except:
            self.avatar_label = tk.Label(self.left_frame, text="Không tìm thấy avatar.png", bg="white", fg="red")
            self.avatar_label.pack()

        # Tên tiếng Việt
        tk.Label(
            self.left_frame,
            text="Lương Ngọc Huy",
            font=("Arial", 14),
            bg="white"
        ).pack(pady=(10, 30))

        # === Bên phải: Chức năng chính ===
        self.btn_choose = tk.Button(
            self.right_frame,
            text="📷 Chọn ảnh",
            command=self.open_image,
            font=("Helvetica", 14),
            bg="#3498db",
            fg="white",
            activebackground="#2980b9",
            activeforeground="white",
            relief=tk.FLAT,
            padx=20,
            pady=5
        )
        self.btn_choose.pack(pady=20)

        # Khu vực hiển thị ảnh được chọn
        self.label_img = tk.Label(
            self.right_frame,
            bg="white",
            bd=1,
            relief="solid"
        )
        self.label_img.pack(pady=10)

        # Kết quả dự đoán
        self.label_result = tk.Label(
            self.right_frame,
            text="",
            font=("Helvetica", 14),
            bg="white",
            fg="#2c3e50"
        )
        self.label_result.pack(pady=10)

    # === Hàm chọn và xử lý ảnh ===
    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            predicted_class, confidence = self.predictor(file_path)

            # Hiển thị ảnh đã chọn
            img = Image.open(file_path).resize((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            self.label_img.config(image=img_tk)
            self.label_img.image = img_tk

            # Hiển thị kết quả
            result_text = f"Prediction: {predicted_class}\nConfidence: {confidence*100:.2f}%"
            self.label_result.config(text=result_text)

