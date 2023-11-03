import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter, load_delegate

# Initialize the TensorFlow Lite interpreter
model_path = 'C:/Users/Matthew/Desktop/Flux_Models/Flux_Stain_Detector.tflite'
interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_path, img_size=(28, 28)):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def run_inference(image_path):
    preprocessed_image = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def select_image():
    global image_path  # Declare 'image_path' as global to modify it

    # Use file dialog to select an image file
    file_path = filedialog.askopenfilename()
    if file_path:
        # Set the global 'image_path' to the selected file path
        image_path = file_path

        # Load and display the image
        img = Image.open(file_path)
        img.thumbnail((200, 200), Image.ANTIALIAS)  # Resize to fit the GUI
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img
        # Update the window title with the image file name
        root.title(f"Flux Stain Detector - {file_path}")


def on_run_inference():
    global image_path  # Access the global variable 'image_path'

    if not image_path:
        result_label.config(text="Please select an image first.")
        return

    result = run_inference(image_path)
    # Update the result label based on the inference
    result_label.config(text=f"Result: {result}")

# Initialize the main window
root = tk.Tk()
root.title("Flux Stain Detector")

# Add a button to select an image
btn_select = tk.Button(root, text="Select Image", command=select_image)
btn_select.pack()

# Add a panel to display the selected image
panel = tk.Label(root)
panel.pack()

# Add a button to run inference
btn_run = tk.Button(root, text="Run Inference", command=on_run_inference)
btn_run.pack()

# Add a label to show the results
result_label = tk.Label(root, text="")
result_label.pack()

# Start the GUI event loop
root.mainloop()
