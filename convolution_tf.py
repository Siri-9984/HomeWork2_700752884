import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Create a directory to save feature maps
os.makedirs("feature_maps", exist_ok=True)

# Define the input matrix
input_matrix = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=np.float32)

# Reshape input to (batch, height, width, channels)
input_tensor = tf.constant(input_matrix.reshape(1, 5, 5, 1))

# Define the kernel
kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float32).reshape(3, 3, 1, 1)

kernel_tensor = tf.constant(kernel)

# Define function to apply convolution and save result
def apply_convolution(input_tensor, kernel_tensor, stride, padding, name):
    output = tf.nn.conv2d(
        input=input_tensor,
        filters=kernel_tensor,
        strides=[1, stride, stride, 1],
        padding=padding
    )
    result = tf.squeeze(output).numpy()
    plt.imshow(result, cmap='viridis')
    plt.title(f'Stride={stride}, Padding={padding}')
    plt.colorbar()
    filename = f'feature_maps/{name}.png'
    plt.savefig(filename)
    plt.show()
    print(f'Saved: {filename}')

# Apply all 4 configurations
apply_convolution(input_tensor, kernel_tensor, 1, 'VALID', 'stride1_valid')
apply_convolution(input_tensor, kernel_tensor, 1, 'SAME', 'stride1_same')
apply_convolution(input_tensor, kernel_tensor, 2, 'VALID', 'stride2_valid')
apply_convolution(input_tensor, kernel_tensor, 2, 'SAME', 'stride2_same')
