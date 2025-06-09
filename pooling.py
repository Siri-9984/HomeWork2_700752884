import tensorflow as tf
import numpy as np

# Create a random 4x4 matrix (1 sample, 4x4 size, 1 channel)
input_matrix = np.random.randint(0, 10, size=(1, 4, 4, 1)).astype(np.float32)

# Define Max Pooling and Average Pooling layers
max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)

# Apply pooling
max_pooled = max_pool(input_matrix)
avg_pooled = avg_pool(input_matrix)

# Print results
print("Original Matrix:\n", input_matrix[0, :, :, 0])
print("\nMax Pooled Matrix:\n", max_pooled.numpy()[0, :, :, 0])
print("\nAverage Pooled Matrix:\n", avg_pooled.numpy()[0, :, :, 0])
