import os
import sys
import sqlite3
import datetime
import threading
import pathlib
import cv2
import queue
import customtkinter as ctk
from PIL import Image
from tkinter import messagebox, filedialog
from engine import ModelEngine
from logger import DriverStateTracker
import time

# DB Setup
DB_NAME = "driver.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS Driver (
                DriverID INTEGER PRIMARY KEY AUTOINCREMENT,
                Name TEXT NOT NULL,
                Password TEXT NOT NULL,
                ContactInfo TEXT,
                LicenseNumber TEXT
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error initializing DB: {e}")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Driver Drowsiness Detection")
        self.geometry("900x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.current_driver = None
        self.session_id = None
        self.log_file = None
        self.tracker = None
        self.video_capture = None
        self.stream_thread = None
        self.is_streaming = False
        self.drowsy_start_time = None
        self._is_playing_sound = False
        
        # Load Model
        self.model_engine = None
        try:
            self.model_engine = ModelEngine()
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Could not load ML Model:\n{e}\nEnsure 'model.h5' is in the directory.")

        init_db()

        self.setup_ui()
        self.bind('<q>', self.stop_stream)
        self.bind('<Q>', self.stop_stream)

    def play_alert_sound(self):
        if getattr(self, '_is_playing_sound', False):
            return
        self._is_playing_sound = True
        def _play():
            sound_path = "/Users/arnavposwal/CascadeProjects/driver_drowsiness_detection/mi-gente-sountec-live-edit.mp3"
            try:
                if sys.platform == "darwin":
                    # macOS natively supports mp3 via afplay
                    os.system(f'afplay "{sound_path}"')
                else:
                    # Windows fallback (winsound doesn't natively support mp3, so we beep)
                    # Alternatively, if we had pygame, we could use that. 
                    import winsound
                    winsound.Beep(1000, 500)
            except ImportError:
                print("\a")
            except Exception as e:
                print(f"Error playing audio: {e}")
            finally:
                self._is_playing_sound = False
        threading.Thread(target=_play, daemon=True).start()

    def setup_ui(self):
        self.tabview = ctk.CTkTabview(self, width=850, height=650)
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)

        self.tab_auth = self.tabview.add("Login / Register")
        self.tab_live = self.tabview.add("Live Detection")
        self.tab_upload = self.tabview.add("Image Classification")

        self.setup_auth_tab()
        self.setup_live_tab()
        self.setup_upload_tab()

    def setup_auth_tab(self):
        self.tab_auth.grid_columnconfigure((0, 1), weight=1)

        # Login Frame
        login_frame = ctk.CTkFrame(self.tab_auth)
        login_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        ctk.CTkLabel(login_frame, text="Login", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)
        self.login_name = ctk.CTkEntry(login_frame, placeholder_text="Name")
        self.login_name.pack(pady=10, padx=20, fill="x")
        self.login_pwd = ctk.CTkEntry(login_frame, placeholder_text="Password", show="*")
        self.login_pwd.pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(login_frame, text="Login", command=self.handle_login).pack(pady=20)

        # Register Frame
        reg_frame = ctk.CTkFrame(self.tab_auth)
        reg_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        ctk.CTkLabel(reg_frame, text="Register", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)
        self.reg_name = ctk.CTkEntry(reg_frame, placeholder_text="Name")
        self.reg_name.pack(pady=10, padx=20, fill="x")
        self.reg_pwd = ctk.CTkEntry(reg_frame, placeholder_text="Password", show="*")
        self.reg_pwd.pack(pady=10, padx=20, fill="x")
        self.reg_contact = ctk.CTkEntry(reg_frame, placeholder_text="Contact Info")
        self.reg_contact.pack(pady=10, padx=20, fill="x")
        self.reg_license = ctk.CTkEntry(reg_frame, placeholder_text="License Number")
        self.reg_license.pack(pady=10, padx=20, fill="x")
        ctk.CTkButton(reg_frame, text="Register", command=self.handle_register).pack(pady=20)

    def setup_live_tab(self):
        self.welcome_label = ctk.CTkLabel(self.tab_live, text="Please login to access live detection.", font=ctk.CTkFont(size=16))
        self.welcome_label.pack(pady=10)

        controls_frame = ctk.CTkFrame(self.tab_live)
        controls_frame.pack(pady=10, fill="x", padx=20)
        
        ctk.CTkLabel(controls_frame, text="EAR Threshold:").pack(side="left", padx=10)
        self.ear_slider = ctk.CTkSlider(controls_frame, from_=0.1, to=0.5, number_of_steps=40, command=self.update_ear_label)
        self.ear_slider.set(0.25)
        self.ear_slider.pack(side="left", fill="x", expand=True, padx=10)
        
        self.ear_value_label = ctk.CTkLabel(controls_frame, text="0.25", width=40)
        self.ear_value_label.pack(side="left", padx=10)

        ctk.CTkLabel(controls_frame, text="Camera ID:").pack(side="left", padx=(20, 5))
        self.camera_id_combo = ctk.CTkComboBox(controls_frame, values=["0", "1"], width=60)
        self.camera_id_combo.set("0")
        self.camera_id_combo.pack(side="left", padx=5)

        self.btn_start_stream = ctk.CTkButton(controls_frame, text="Start Stream", command=self.toggle_stream, state="disabled")
        self.btn_start_stream.pack(side="right", padx=10)

        self.video_label = ctk.CTkLabel(self.tab_live, text="Camera feed will appear here")
        self.video_label.pack(pady=10, fill="both", expand=True)

    def update_ear_label(self, value):
        self.ear_value_label.configure(text=f"{value:.2f}")

    def setup_upload_tab(self):
        ctk.CTkLabel(self.tab_upload, text="Upload an Image for Classification", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=20)
        self.upload_img_label = ctk.CTkLabel(self.tab_upload, text="No image selected", width=400, height=300)
        self.upload_img_label.pack(pady=10)

        self.res_label = ctk.CTkLabel(self.tab_upload, text="", font=ctk.CTkFont(size=16))
        self.res_label.pack(pady=10)

        ctk.CTkButton(self.tab_upload, text="Select Image", command=self.upload_image).pack(pady=10)

    def handle_login(self):
        name = self.login_name.get().strip()
        pwd = self.login_pwd.get().strip()
        if not name or not pwd:
            messagebox.showwarning("Login", "Please enter name and password.")
            return

        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("SELECT * FROM Driver WHERE Name=? AND Password=?", (name, pwd))
            row = c.fetchone()
            conn.close()

            if row:
                self.current_driver = {"DriverID": row[0], "Name": row[1]}
                session_start = datetime.datetime.now()
                self.session_id = session_start.strftime("%Y%m%d_%H%M%S")
                logs_dir = os.path.join("logs", name, self.session_id)
                os.makedirs(logs_dir, exist_ok=True)
                self.log_file = os.path.join(logs_dir, "session_log.txt")
                self.tracker = DriverStateTracker(self.log_file)

                messagebox.showinfo("Login", f"Welcome, {name}!")
                self.welcome_label.configure(text=f"Welcome, {name}! Monitor your drowsiness below.")
                self.btn_start_stream.configure(state="normal")
                self.tabview.set("Live Detection")
            else:
                messagebox.showerror("Login Error", "Invalid credentials.")
        except Exception as e:
            messagebox.showerror("DB Error", str(e))

    def handle_register(self):
        name = self.reg_name.get().strip()
        pwd = self.reg_pwd.get().strip()
        contact = self.reg_contact.get().strip()
        lic = self.reg_license.get().strip()
        if not name or not pwd:
            messagebox.showwarning("Register", "Name and Password are required.")
            return

        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("INSERT INTO Driver (Name, Password, ContactInfo, LicenseNumber) VALUES (?, ?, ?, ?)",
                      (name, pwd, contact, lic))
            conn.commit()
            conn.close()
            messagebox.showinfo("Success", "Registration successful! You can now log in.")
        except Exception as e:
            messagebox.showerror("Register Error", str(e))

    def stop_stream(self, event=None):
        if not self.is_streaming:
            return
        self.is_streaming = False
        self.btn_start_stream.configure(text="Start Stream")
        self.video_label.configure(image="", text="Camera stopped")
        self.drowsy_start_time = None

    def toggle_stream(self):
        if not self.model_engine:
            messagebox.showerror("Error", "Model not loaded properly.")
            return

        if self.is_streaming:
            self.stop_stream()
        else:
            cam_idx = int(self.camera_id_combo.get())
            self.video_capture = cv2.VideoCapture(cam_idx)
            
            if not getattr(self, 'video_capture', None) or not self.video_capture.isOpened():
                messagebox.showerror(
                    "Camera Error",
                    f"Could not open camera ID {cam_idx}.\n\n"
                    "If using a wired iPhone (Continuity Camera), it might be on ID 1 or 2 instead of 0.\n"
                    "Also ensure your Terminal has permission in System Settings > Privacy & Security > Camera."
                )
                return
            self.is_streaming = True
            self.btn_start_stream.configure(text="Stop Stream")
            
            # Start streaming thread
            self.stream_thread = threading.Thread(target=self.stream_loop, daemon=True)
            self.stream_thread.start()

    def stream_loop(self):
        while self.is_streaming:
            ret, frame = self.video_capture.read()
            if not ret:
                self.after(0, self.stop_stream)
                break
            
            # Process frame using the ML engine
            try:
                ear_thresh = self.ear_slider.get()
                processed_frame, is_drowsy, metrics = self.model_engine.detect_drowsiness_live(frame.copy(), ear_threshold=ear_thresh)
                
                avg_ear = metrics.get('avg_ear', 0.0)
                mouth_dist = metrics.get('mouth_dist', 0.0)
                face_detected = metrics.get('face_detected', False)
                h, w, _ = frame.shape
                is_yawning = mouth_dist > (0.08 * h)

                if self.tracker:
                    self.tracker.update_state_v2(ear_thresh, face_detected, is_yawning, metrics)
                
                # Check tracker's official state to determine if we should play sound.
                # Rely on tracker instead of app.py timer, which guarantees synchronized logs & UI sound.
                is_officially_drowsy = (self.tracker and self.tracker.current_state.value == "DROWSY")
                
                if is_officially_drowsy:
                    self.play_alert_sound()
                    
            except Exception as e:
                print(f"Frame processing error: {e}")
                if self.tracker:
                    self.tracker.log_error(f"Frame processing error: {e}")
                processed_frame = frame
            
            # Convert BGR to RGB for PIL/Tkinter
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(processed_frame_rgb)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            
            # Update UI on main thread safely
            self.after(0, self.update_video_frame, ctk_img)
            
        # Clean up capture when loop ends
        if getattr(self, 'video_capture', None):
            self.video_capture.release()
            self.video_capture = None

    def update_video_frame(self, ctk_img):
        self.video_label.configure(image=ctk_img, text="")

    def upload_image(self):
        if not self.model_engine:
            messagebox.showerror("Error", "Model not loaded.")
            return

        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Could not read image.")
            return

        # Display uploaded image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((400, 300))
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
        self.upload_img_label.configure(image=ctk_img, text="")

        # Predict using a background thread to prevent UI freezing
        self.res_label.configure(text="Classifying...", text_color="white")
        threading.Thread(target=self.classify_task, args=(img_rgb,), daemon=True).start()

    def classify_task(self, img_rgb):
        try:
            # Haar Cascade extraction is usually done in the engine if we want to isolate face
            # But the original code was doing custom face extraction before Model.predict
            # I will just pass the raw frame and let engine handle resizing to 224x224
            label, conf = self.model_engine.predict_image(img_rgb)
            text_color = "red" if label == "Drowsy" else "green"
            self.after(0, lambda: self.res_label.configure(text=f"{label} ({conf:.2f}%)", text_color=text_color))
        except Exception as e:
            self.after(0, lambda: self.res_label.configure(text=f"Error: {e}", text_color="red"))

    def on_closing(self):
        self.is_streaming = False
        
        # Wait for the stream thread to cleanly release the camera to prevent segfaults
        if getattr(self, 'stream_thread', None) and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2.0)
            
        if getattr(self, 'tracker', None):
            self.tracker.end_session()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
