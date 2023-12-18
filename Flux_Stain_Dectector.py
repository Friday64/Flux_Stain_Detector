import sys
import threading
from queue import Queue, Empty
import cv2
import numpy as np
import torch
import torch.onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import tkinter as tk
from PIL import Image, ImageTk
from torchvision import transforms

# Global variables
detection_active = False
frame_queue = Queue(maxsize=5)

# Initialize the camera with 1080p resolution
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Function to convert PyTorch model to ONNX and then to TensorRT Engine
def convert_model_to_trt_engine(model, input_size, onnx_file_path, trt_engine_path):
    # Convert PyTorch model to ONNX
    model.eval()
    dummy_input = torch.randn(1, *input_size).to('cuda')
    torch.onnx.export(model, dummy_input, onnx_file_path, verbose=True)

    # Convert ONNX model to TensorRT Engine
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    engine = builder.build_engine(network, config)

    with open(trt_engine_path, "wb") as f:
        f.write(engine.serialize())

    return engine

# TensorRT Inference Class
class TRTInference:
    def __init__(self, engine_file_path):
        self.engine = self.load_engine(engine_file_path)
        self.context = self.engine.create_execution_context()

    def load_engine(self, engine_file_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def infer(self, input_data):
        # Allocate buffers and perform inference
        # This needs to be implemented based on your model's input and output
        # ...

# Initialize TensorRT inference
trt_inference = TRTInference('path_to_your_trt_model.trt')

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
                    raw_detection_results = trt_inference.infer(preprocessed_frame)
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
