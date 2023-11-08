import tkinter as tk
from tkinter import Frame, Label
import cv2
from PIL import Image, ImageTk

# Create a window
window = tk.Tk()
window.title("Flux Stain Detector")

# Create a frame for the video feed
video_frame = Frame(window)
video_frame.pack(padx=10, pady=10)

# Label for displaying the video frames
video_label = Label(video_frame)
video_label.pack()

# Function to update the video label with the latest frame
def update_video_label(frame):
    # Convert the image to a format that Tkinter can use
    cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image)
    tk_image = ImageTk.PhotoImage(image=pil_image)
    
    # Update the label with the new image
    video_label.imgtk = tk_image
    video_label.configure(image=tk_image)

# Initialize the USB webcam feed
camera = cv2.VideoCapture(0)

# Set the resolution of the video
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def show_frame():
    # Capture frame-by-frame
    ret, frame = camera.read()

    if ret:
        # Here you would add your model prediction and annotation code

        # Update the GUI with the new frame
        update_video_label(frame)

    # Call this function again after a short delay to update the video feed
    window.after(10, show_frame)

# Start the frame update loop
show_frame()

# Run the GUI main loop
window.mainloop()

# Release the video capture when the GUI is closed
camera.release()
