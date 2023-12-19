import threading
from queue import Queue, Empty
import cv2
import tensorflow as tf
import numpy as np
import torch
import torch.onnx
import tkinter as tk
from PIL import Image, ImageTk
from torchvision import transforms
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt


def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
    

def trt_inference(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
    
def preprocess_frame(frame):
    # Apply image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(frame)


def postprocess_output(output_data):
    # Example post-processing: Convert Tensor to Numpy array
    results = output_data.cpu().numpy()
    return results


def process_detection_results(raw_detection_results):
    # Example detection processing: Convert detection results to boolean array
    detection_results = raw_detection_results > 0.5
    return detection_results


def draw_detection_results(frame, detection_results):
    # Example detection visualization: Draw detection results on the frame
    frame = cv2.putText(frame, 'Flux Stain Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return frame



# Initialize CUDA context (make sure to set the correct CUDA device)
cuda.init()
device = cuda.Device(0)  # Change the device index if needed
context = device.make_context()

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
    network = builder.create_network(trt.EXPLICIT_BATCH)
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

        # Define the size of the output buffer based on your model's output size
        self.output_size = self.engine.get_binding_shape(1)[-1]  # Replace with the actual size

        # Allocate memory for the output data
        self.output_data = np.empty((self.output_size,), dtype=np.float32)  # Adjust data type as needed

    def load_engine(self, engine_file_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def infer(self, input_data):
        # Define the input and output bindings for the TensorRT engine
        input_binding_idx = self.engine.get_binding_index('input')  # Adjust as needed
        output_binding_idx = self.engine.get_binding_index('output')  # Adjust as needed

        # Prepare input data (e.g., convert to suitable format, resize, normalize, etc.)
        preprocessed_input = preprocess_input(input_data)  # Implement this function

        # Allocate input and output buffers on GPU
        input_buffer = cuda.mem_alloc(preprocessed_input.nbytes)  # Adjust for your input data type
        output_buffer = cuda.mem_alloc(self.output_size * 4)  # 4 bytes per float32

        # Transfer input data to GPU memory
        cuda.memcpy_htod(input_buffer, preprocessed_input)

        # Execute inference
        self.context.execute(bindings=[int(input_buffer), int(output_buffer)])

        # Transfer output data from GPU to CPU
        cuda.memcpy_dtoh(self.output_data, output_buffer)

        # Post-process output data and return the results
        results = postprocess_output(self.output_data)  # Implement this function
        return results

# Example of preprocess_input and postprocess_output functions (customize for your model)
def preprocess_input(input_data):
    # Resize and normalize input image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    preprocessed_input = transform(input_data)
    preprocessed_input = preprocessed_input.unsqueeze(0)  # Add batch dimension
    return preprocessed_input

def postprocess_output(output_data):
    # Example post-processing: Convert Tensor to Numpy array
    results = output_data.cpu().numpy()
    return results

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

if __name__ == "__main__":
    app = FluxStainDetectorApp()
    app.mainloop()
