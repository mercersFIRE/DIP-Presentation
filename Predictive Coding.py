import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function to apply Run-Length Encoding (RLE) for compression
def rle_compress(data):
    compressed_data = []
    current_value = data[0]
    count = 1
    for symbol in data[1:]:
        if symbol == current_value:
            count += 1
        else:
            compressed_data.append((current_value, count))
            current_value = symbol
            count = 1
    compressed_data.append((current_value, count))  # Append the last sequence
    return compressed_data

# Run-Length Encoding (RLE) Decompression
def rle_decompress(compressed_data):
    decompressed_data = []
    for symbol, count in compressed_data:
        decompressed_data.extend([symbol] * count)
    return np.array(decompressed_data)

# Predictive Coding function (with left pixel prediction)
def predictive_coding(image_path):
    img = Image.open(image_path).convert("L")  # Convert image to grayscale
    img_data = np.array(img)
    
    # Predict the image data based on the left pixel (excluding the first column)
    predicted_data = np.zeros_like(img_data)
    residual_data = np.zeros_like(img_data, dtype=np.int16)  # Use int16 for residuals to avoid overflow
    
    # Predict based on the left neighbor
    for i in range(1, img_data.shape[0]):  # Skip the first row
        for j in range(1, img_data.shape[1]):  # Skip the first column
            predicted_data[i, j] = img_data[i, j-1]  # Predict from the left pixel
            residual_data[i, j] = img_data[i, j] - predicted_data[i, j]  # Calculate residual
            
            # Clip residuals to stay within the valid range of uint8 (0-255)
            residual_data[i, j] = np.clip(residual_data[i, j], -128, 127)  # Clipping for int16

    # Flatten the residual data and compress it using RLE
    residual_data_flattened = residual_data.flatten()
    compressed_residuals = rle_compress(residual_data_flattened)
    
    # Decompress the residuals using RLE
    decompressed_residuals = rle_decompress(compressed_residuals)
    
    # Reconstruct the image using the predicted values and decompressed residuals
    decompressed_img_data = np.zeros_like(img_data)
    idx = 0
    for i in range(1, img_data.shape[0]):  # Skip the first row
        for j in range(1, img_data.shape[1]):  # Skip the first column
            decompressed_img_data[i, j] = predicted_data[i, j] + decompressed_residuals[idx]
            idx += 1
    
    decompressed_image = Image.fromarray(decompressed_img_data.astype(np.uint8))
    
    return img, decompressed_image, img.size, decompressed_image.size

# Main function
image_path = "input_image.jpg"  # Replace with your image path
original_img, decompressed_img, original_size, decompressed_size = predictive_coding(image_path)

original_size_mb = original_size[0] * original_size[1] / (1024 * 1024)  # Convert to MB
decompressed_size_mb = decompressed_size[0] * decompressed_size[1] / (1024 * 1024)  # Convert to MB

print(f"Original Image Size: {original_size_mb:.2f} MB")
print(f"Decompressed Image Size: {decompressed_size_mb:.2f} MB")

# Show results
plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 2, 1)
plt.title(f"Original Image\nSize: {original_size_mb:.2f} MB")
plt.imshow(original_img, cmap="gray")

# Decompressed image
plt.subplot(1, 2, 2)
plt.title(f"Decompressed Image\nSize: {decompressed_size_mb:.2f} MB")
plt.imshow(decompressed_img, cmap="gray")

plt.show()
