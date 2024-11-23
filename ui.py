import tkinter as tk
from PIL import Image, ImageGrab, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('digit_recognition_model.h5')

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")
    result_label.config(text="Predicted Digit: ")

# Function to predict the drawn digit
def predict_digit():
    try:
        # Save the canvas drawing to an image
        x0 = root.winfo_rootx() + canvas.winfo_x()
        y0 = root.winfo_rooty() + canvas.winfo_y()
        x1 = x0 + canvas.winfo_width()
        y1 = y0 + canvas.winfo_height()
        image = ImageGrab.grab().crop((x0, y0, x1, y1))
        
        # Preprocess the image
        image = image.convert('L')  # Convert to grayscale
        image = ImageOps.invert(image)  # Invert colors (white becomes black)
        image = image.resize((28, 28))  # Resize to 28x28
        image_array = np.array(image).reshape(1, 28, 28, 1) / 255.0  # Normalize
        
        # Apply thresholding to make the image more distinct
        image_array[image_array > 0.5] = 1
        image_array[image_array <= 0.5] = 0
        
        # Predict the digit
        prediction = model.predict(image_array)
        digit = np.argmax(prediction)
        
        # Display the result
        result_label.config(text=f"Predicted Digit: {digit}")
    
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")

# Create the main application window
root = tk.Tk()
root.title("Digit Recognizer")

# Create the canvas for drawing
canvas = tk.Canvas(root, width=280, height=280, bg="white")
canvas.grid(row=0, column=0, columnspan=2)

# Add buttons and labels
clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.grid(row=1, column=0)

predict_button = tk.Button(root, text="Predict", command=predict_digit)
predict_button.grid(row=1, column=1)

result_label = tk.Label(root, text="Predicted Digit: ", font=("Helvetica", 14))
result_label.grid(row=2, column=0, columnspan=2)

# Enable drawing on the canvas
def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")

canvas.bind("<B1-Motion>", paint)

# Run the application
root.mainloop()





