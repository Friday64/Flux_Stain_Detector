# Built-in Libraries
import sys
import threading
from queue import Queue, Empty

# Non-built-in Libraries
import cv2
import numpy as np
import torch  # PyTorch
from torchvision import transforms
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QImage

# Global variables
detection_active = False
model = None
model_loading_thread = None
frame_queue = Queue(maxsize=20)

# Initialize the camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Function to load the PyTorch model in a separate thread
def load_model_threaded():
    global model
    model_path = 'C:/Users/Matthew/Desktop/Programming/Detect_Flux_Project/Flux_Models'
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    model.cuda()  # Move model to GPU
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
                preprocessed_frame = preprocessed_frame.cuda()  # Move data to GPU
                with torch.no_grad():
                    raw_detection_results = model(preprocessed_frame)

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
    # Replace with your model's expected preprocessing
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
    # Replace with your model's specific processing
    raw_results = raw_results.cpu().numpy()  # Move data back to CPU
    confidence_scores = raw_results[:, 0]
    return np.any(confidence_scores > confidence_threshold)

# PyQt5 Application Class
class FluxStainDetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Flux Stain Detector')
        self.setGeometry(300, 300, 350, 250)  # Modify as needed
        layout = QVBoxLayout()
        
        self.video_label = QLabel(self)
        layout.addWidget(self.video_label)

        self.start_btn = QPushButton('Start Detection', self)
        self.start_btn.clicked.connect(self.startDetection)
        layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton('Stop Detection', self)
        self.stop_btn.clicked.connect(self.stopDetection)
        layout.addWidget(self.stop_btn)

        self.setLayout(layout)

    def startDetection(self):
        global detection_active
        detection_active = True
        threading.Thread(target=camera_capture_thread).start()
        threading.Thread(target=detection_thread, args=(self,)).start()

    def stopDetection(self):
        global detection_active
        detection_active = False

    def update_video_label(self, frame):
        qformat = QImage.Format_Indexed8 if len(frame.shape) == 2 else QImage.Format_RGB888
        out_image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
        out_image = out_image.rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(out_image))

if __name__ == "__main__":
    app = QApplication([])
    window = FluxStainDetectorApp()
    window.show()
    sys.exit(app.exec_())
