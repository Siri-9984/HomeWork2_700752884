import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread(r'c:\Users\91630\OneDrive\Desktop\Home_Assignment2_700752884\image1.jpg', cv2.IMREAD_GRAYSCALE)

# Define Sobel filters
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

# Apply convolution using filter2D
edge_x = cv2.filter2D(image, -1, sobel_x)
edge_y = cv2.filter2D(image, -1, sobel_y)

# Save the results instead of displaying
plt.imsave("original_image.png", image, cmap='gray')
plt.imsave("sobel_x_result.png", edge_x, cmap='gray')
plt.imsave("sobel_y_result.png", edge_y, cmap='gray')

print("Images saved: original_image.png, sobel_x_result.png, sobel_y_result.png")
