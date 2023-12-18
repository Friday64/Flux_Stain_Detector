import sys
import threading
from queue import Queue, Empty
import cv2
import numpy as np
import torch
from torchvision import transforms
import tkinter as tk
from PIL import Image, ImageTk

# Global variables
detection_active = False
model = None
frame_queue = Queue(maxsize=5)  # Reduced queue size for memory efficiency

# Initialize the camera with reduced resolution for better performance
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load the PyTorch model in a separate thread
def load_model_threaded():
    global model
    model_path = 'C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Models'
    model = torch.load(model_path)
    model.eval()
    model.to('cuda')  # Ensure model is on GPU
    print("Model loaded successfully.")

# Camera capture thread
def camera_capture_thread():
    global detection_active
    while detection_active:
        ret, frame = camera.read()
        if ret:
            if not frame_queue.full():
                frame_queue.put(frame)
            else:
                print("Frame buffer is full, dropping frame")

# Detection thread
def detection_thread(app):
    global detection_active
    while detection_active:
        try:
            frame = frame_queue.get(timeout=1)
            if detection_active and frame is not None:
                preprocessed_frame = preprocess_frame(frame)
                preprocessed_frame = preprocessed_frame.to('cuda')
                with torch.no_grad():
                    raw_detection_results = model(preprocessed_frame)
                stain_detected = process_detection_results(raw_detection_results)
                if stain_detected.any():
                    cv2.putText(frame, 'Flux Stain Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 2, cv2.LINE_AA)
                app.update_video_label(frame)
        except Empty:
            continue

# Preprocess frame
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Example normalization
    ])
    frame = transform(frame)
    frame = frame.unsqueeze(0)  # Add batch dimension
    return frame

# Process detection results
def process_detection_results(raw_results, confidence_threshold=0.5):
    raw_results = raw_results.cpu().numpy()  # Move data back to CPU
    confidence_scores = raw_results[:, 0]
    return np.any(confidence_scores > confidence_threshold)

# Tkinter Application Class
class FluxStainDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Flux Stain Detector')
        self.geometry('800x600')  # Adjusted for reduced frame size

        self.video_label = tk.Label(self)
        self.video_label.pack()

        self.start_btn = tk.Button(self, text='Start Detection', command=self.startDetection)
        self.start_btn.pack()

        self.stop_btn = tk.Button(self, text='Stop Detection', command=self.stopDetection)
        self.stop_btn.pack()

    def startDetection(self):
        global detection_active
        detection_active = True
        threading.Thread(target=camera_capture_thread).start()
        threading.Thread(target=detection_thread, args=(self,)).start()

    def stopDetection(self):
        global detection_active
        detection_active = False
        camera.release()  # Release the camera resource

    def update_video_label(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

if __name__ == "__main__":
    app = FluxStainDetectorApp()
    app.mainloop()
