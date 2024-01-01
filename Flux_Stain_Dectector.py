import threading
from queue import Queue, Empty
import cv2
import torch
from torchvision import transforms
import tkinter as tk
from PIL import Image, ImageTk

# Global variables
detection_active = False
frame_queue = Queue(maxsize=5)

# Model path
model_path = "path/to/your/model.pth"  # Update with your actual model path

# Initialize the camera with 1080p resolution and 60 FPS
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
camera.set(cv2.CAP_PROP_FPS, 60)  # Attempt to set camera FPS to 60

# Load PyTorch Model
# Ensure the model is compatible and optimized for Jetson Nano
model = torch.load(model_path)
model = model.cuda()  # Move model to GPU if CUDA is available
model.eval()

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
                tensor_frame = preprocess_frame(frame)
                with torch.no_grad():
                    detection_results = model(tensor_frame)
                # Update frame with any overlays or results
                app.update_video_label(frame)  
        except Empty:
            continue

# Preprocess frame
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    frame = Image.fromarray(frame)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Example resize, adjust to your needs
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pre-trained models
    ])
    frame = transform(frame)
    frame = frame.unsqueeze(0)  # Add batch dimension
    frame = frame.cuda()  # Move to GPU if CUDA is available
    return frame

# Tkinter Application Class
class FluxStainDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Flux Stain Detector')
        self.geometry('800x600')

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
        camera.release()

    def update_video_label(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

if __name__ == "__main__":
    app = FluxStainDetectorApp()
    app.mainloop()
