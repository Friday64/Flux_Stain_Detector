import tkinter as tk
from tkinter import Frame, Label, Button
import cv2
from PIL import Image, ImageTk
import tensorflow as tf

# Load your trained TensorFlow model
model = tf.keras.models.load_model('path_to_your_model.h5')

# Initialize the USB webcam feed
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create a window
window = tk.Tk()
window.title("Flux Stain Detector")

# Create a frame for the video feed
video_frame = Frame(window)
video_frame.pack(padx=10, pady=10)

# Label for displaying the video frames
video_label = Label(video_frame)
video_label.pack()

# Function to preprocess the frame before feeding it to the model
def preprocess_frame(frame):
    # Resize and normalize the frame to match the model's expected input
    frame = cv2.resize(frame, (28, 28))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame / 255.0
    frame = frame.reshape((1, 28, 28, 1))
    return frame

# Function to annotate the frame with detection results
def annotate_frame(frame, detection_results):
    # Process the detection results and annotate the frame
    # This is a placeholder, you'll need to implement the logic based on your model's output
    return frame

# Function to update the video label with the latest frame
def update_video_label(frame):
    # Convert the image to a format that Tkinter can use
    cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image)
    tk_image = ImageTk.PhotoImage(image=pil_image)
    
    # Update the label with the new image
    video_label.imgtk = tk_image
    video_label.configure(image=tk_image)

# Function to handle the detection process
def detect_stains():
    # Capture frame-by-frame
    ret, frame = camera.read()

    if ret:
        # Preprocess the frame for the model
        preprocessed_frame = preprocess_frame(frame)

        # Use the model to detect flux stains in the frame
        detection_results = model.predict(preprocessed_frame)

        # Annotate the frame with the detection results
        annotated_frame = annotate_frame(frame, detection_results)

        # Update the GUI with the new frame
        update_video_label(annotated_frame)

    # Call this function again after a short delay to update the video feed
    window.after(10, detect_stains)

# Start and stop buttons
start_button = Button(window, text="Start", command=detect_stains)
start_button.pack(side=tk.LEFT, padx=5, pady=5)

stop_button = Button(window, text="Stop", command=lambda: window.quit())
stop_button.pack(side=tk.LEFT, padx=5, pady=5)

# Run the GUI main loop
window.mainloop()

# Release the video capture when the GUI is closed
camera.release()