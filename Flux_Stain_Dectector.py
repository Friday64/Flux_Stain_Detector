import sys
import cv2
import numpy as np
import tensorflow as tf
import threading
from queue import Queue, Empty
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QComboBox
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer

# Global variables
detection_active = False
model = None
model_loading_thread = None
frame_queue = Queue(maxsize=20)

# Initialize the camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Function to load the TensorFlow model in a separate thread
def load_model_threaded():
    global model
    model_path = 'C:/Users/Matthew/Desktop/Programming/Flux_Models/flux_model.h5'
    model = tf.keras.models.load_model(model_path)
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
                # Preprocess the frame for the model
                preprocessed_frame = preprocess_frame(frame)

                # Use the model to detect flux stains in the frame
                raw_detection_results = model.predict(preprocessed_frame)

                # Process the detection results
                stain_detected = process_detection_results(raw_detection_results)

                if stain_detected.any():
                    cv2.putText(frame, 'Flux Stain Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 2, cv2.LINE_AA)

                # Update the GUI with the new frame
                app.update_video_label(frame)
        except Empty:
            continue

# Preprocess frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (28, 28))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame / 255.0
    frame = frame.reshape((1, 28, 28, 1))
    return frame

# Process detection results
def process_detection_results(raw_results, confidence_threshold=0.5):
    confidence_scores = raw_results[:, 0]
    return np.any(confidence_scores > confidence_threshold)

# Main Application Window Class
class FluxStainDetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Flux Stain Detector")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout(self)
        self.video_label = QLabel(self)
        layout.addWidget(self.video_label)

        self.camera_dropdown = QComboBox(self)
        self.camera_dropdown.addItems([str(i) for i in get_available_cameras()])
        layout.addWidget(self.camera_dropdown)

        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_detection)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_detection)
        layout.addWidget(self.stop_button)

        self.setLayout(layout)

    def start_detection(self):
        global detection_active, model_loading_thread, camera_thread, detection_thread
        detection_active = True
        camera_thread = threading.Thread(target=camera_capture_thread, daemon=True)
        detection_thread = threading.Thread(target=detection_thread, args=(self,), daemon=True)
        camera_thread.start()
        detection_thread.start()
        print("Detection started.")

    def stop_detection(self):
        global detection_active
        detection_active = False
        camera.release()

    def update_video_label(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_qt_format)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height()))

def get_available_cameras(limit=10):
    available_cameras = []
    for i in range(limit):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

if __name__ == "__main__":
    app = QApplication([])
    window = FluxStainDetectorApp()
    window.show()
    sys.exit(app.exec())
