import tkinter as tk
from tkinter import Frame, Label, Button, ttk
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
import threading
import numpy as np
from queue import Queue, Empty

# Global variables
detection_active = False
refresh_rate = int(1000 / 60)  # Refresh rate in milliseconds for 60Hz
model = None
model_loading_thread = None
frame_queue = Queue(maxsize=10)

# Initialize the camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def camera_capture_thread():
    global detection_active
    while detection_active:
        ret, frame = camera.read()
        if ret:
            if not frame_queue.full():
                frame_queue.put(frame)
            else:
                print("Frame buffer is full, dropping frame")
def detection_thread():
    global detection_active
    while detection_active:
        try:
            frame = frame_queue.get(timeout=1)  # Adjust timeout as needed
            # [Detection logic goes here, similar to what was in detect_stains]
        except Empty:
            continue
# Function to load the TensorFlow model in a separate thread
def load_model_threaded():
    global model
    model_path = 'C:/Users/Matthew/Desktop/Flux_Models/flux_model.h5'
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

# Function definitions
def get_available_cameras(limit=10):
    available_cameras = []
    for i in range(limit):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def switch_camera_source(camera_index):
    global camera
    try:
        print(f"Switching to camera index: {camera_index}")
        if camera.isOpened():
            camera.release()  # Release the current camera
        camera = cv2.VideoCapture(camera_index)  # Open the new camera
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not camera.isOpened():
            raise ValueError(f"Camera with index {camera_index} could not be opened.")
        else:
            print("Camera switched successfully.")
    except Exception as e:
        print(f"Failed to switch the camera. Error: {e}")

def on_camera_select(event):
    camera_index = int(camera_dropdown.get())
    threading.Thread(target=switch_camera_source, args=(camera_index,), daemon=True).start()
    
# Function to start detection
def start_detection():
    global detection_active, model_loading_thread, camera_thread, detection_thread
    # [Existing code for model loading]
    detection_active = True
    camera_thread = threading.Thread(target=camera_capture_thread, daemon=True)
    detection_thread = threading.Thread(target=detection_thread, daemon=True)
    camera_thread.start()
    detection_thread.start()
    print("Detection started.")


def stop_detection():
    global detection_active
    detection_active = False
    camera.release()  # release the camera when stopping detection
def preprocess_frame(frame):
    frame = cv2.resize(frame, (28, 28))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame / 255.0
    frame = frame.reshape((1, 28, 28, 1))
    return frame

def annotate_frame(frame, detection_results, threshold=0.8):
    # If any detection result exceeds the threshold, we'll add a label to the frame
    for confidence in detection_results:
        if confidence > threshold:
            # Draw the label on the frame
            label_position = (10, 30)  # Position the text at the top-left corner of the frame
            cv2.putText(frame, 'Flux Stain Detected', label_position, cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            break  # Since we only need one label, we can break after adding it
    return frame

def update_video_label(frame):
    cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image)
    tk_image = ImageTk.PhotoImage(image=pil_image)
    video_label.imgtk = tk_image
    video_label.configure(image=tk_image)

def detect_stains():
    # Capture frame-by-frame
    ret, frame = camera.read()

    if detection_active and ret:
        # Preprocess the frame for the model
        preprocessed_frame = preprocess_frame(frame)

        # Use the model to detect flux stains in the frame
        raw_detection_results = model.predict(preprocessed_frame)

        # Process the detection results
        stain_detected = process_detection_results(raw_detection_results)

        if stain_detected.any():  # Changed to use the .any() method
            # Add a label or change color, etc.
            cv2.putText(frame, 'Flux Stain Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        # Update the GUI with the new frame
        update_video_label(frame)

    # Call this function again after a short delay to update the video feed
    window.after(refresh_rate, detect_stains)



def process_detection_results(raw_results, confidence_threshold=0.5):
    # If raw_results is a multi-dimensional array, you might need to adjust
    # how you're accessing the confidence scores.
    # Assuming raw_results is an array where each row corresponds to a prediction
    # and the first column contains the confidence score:
    confidence_scores = raw_results[:, 0]  # This will extract all confidence scores

    # Check if any of the confidence scores exceed the threshold
    return np.any(confidence_scores > confidence_threshold)

# GUI setup
window = tk.Tk()
window.title("Flux Stain Detector")

video_frame = Frame(window)
video_frame.pack(padx=10, pady=10)

video_label = Label(video_frame)
video_label.pack()

camera_options = get_available_cameras()
camera_dropdown = ttk.Combobox(window, values=camera_options, state="readonly")
camera_dropdown.pack(side=tk.LEFT, padx=5, pady=5)
camera_dropdown.current(0)  # Set default camera
camera_dropdown.bind('<<ComboboxSelected>>', on_camera_select)

start_button = Button(window, text="Start", command=start_detection)  # Set command to start_detection
start_button.pack(side=tk.LEFT, padx=5, pady=5)

stop_button = Button(window, text="Stop", command=stop_detection)  # Set command to stop_detection
stop_button.pack(side=tk.LEFT, padx=5, pady=5)


# Main loop
window.mainloop()

# Cleanup on close
if camera.isOpened():
    camera.release()
