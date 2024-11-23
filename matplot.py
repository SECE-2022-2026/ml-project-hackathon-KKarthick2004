import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Display some sample images from the training dataset
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    axes[i].imshow(x_train[i], cmap='gray')  # Show image in grayscale
    axes[i].set_title(f"Label: {y_train[i]}")  # Display the digit label
    axes[i].axis('off')  # Hide axes
plt.show()
