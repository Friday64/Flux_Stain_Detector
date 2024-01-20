import tensorflow as tf
from tflite_runtime.interpreter import load_delegate
import threading
from queue import Queue, Empty
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

def convert_model_to_tflite(model_path, output_path):
    """
    Convert a TensorFlow model to a TensorFlow Lite model.

    :param model_path: Path to the original TensorFlow model.
    :param output_path: Path where the TensorFlow Lite model will be saved.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Ensure model is compatible with Coral (e.g., quantization)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Model converted and saved to {output_path}")
    except Exception as e:
        print(f"Error during model conversion: {e}")

# Global variables
detection_active = False
frame_queue = Queue(maxsize=5)

# Model path
model_path = "path/to/your/model.tflite"  # Update with your actual TensorFlow Lite model path

# Load TFLite model and allocate tensors
try:
    interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize the camera with 1080p resolution and 60 FPS
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
camera.set(cv2.CAP_PROP_FPS, 60)

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
                try:
                    interpreter.set_tensor(input_details[0]['index'], tensor_frame)
                    interpreter.invoke()
                    detection_results = interpreter.get_tensor(output_details[0]['index'])
                    # Additional processing of results can be added here
                    app.update_video_label(frame)
                except Exception as e:
                    print(f"Error during model inference: {e}")
        except Empty:
            continue

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

# Example usage of the conversion function
# Note: Uncomment the following lines to convert a model before using it in the application.
# original_model_path = 'path/to/your/original/tensorflow/model'
# tflite_output_path = '
