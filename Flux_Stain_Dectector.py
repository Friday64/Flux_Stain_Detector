import tensorflow as tf
from tflite_runtime.interpreter import Interpreter
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Global variable
detection_active = False

# Model path
model_path = "path/to/your/model.tflite"  # Update with your actual TensorFlow Lite model path

# Load TFLite model and allocate tensors
try:
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize the camera with 1080p resolution
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Preprocess frame
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    frame = np.expand_dims(frame, axis=0)
    frame = frame.astype(np.uint8)
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
        self.detect_stains()

    def stopDetection(self):
        global detection_active
        detection_active = False
        camera.release()

    def detect_stains(self):
        global detection_active
        while detection_active:
            ret, frame = camera.read()
            if ret:
                tensor_frame = preprocess_frame(frame)
                try:
                    interpreter.set_tensor(input_details[0]['index'], tensor_frame)
                    interpreter.invoke()
                    detection_results = interpreter.get_tensor(output_details[0]['index'])
                    # Additional processing of results can be added here
                    self.update_video_label(frame)
                except Exception as e:
                    print(f"Error during model inference: {e}")

    def update_video_label(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

# Run the application
if __name__ == '__main__':
    app = FluxStainDetectorApp()
    app.mainloop()
