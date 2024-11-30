import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Golomb encoding
def golomb_encode(data, m):
    encoded_data = []
    for value in data:
        quotient = value // m
        remainder = value % m
        unary_code = '1' * quotient + '0'
        remainder_code = f"{remainder:08b}"  # Fixed 8-bit binary representation
        encoded_data.append(unary_code + remainder_code)
    return ''.join(encoded_data)

# Golomb decoding
def golomb_decode(encoded_data, m):
    decoded_data = []
    idx = 0
    n = len(encoded_data)

    while idx < n:
        # Decode unary part
        quotient = 0
        while idx < n and encoded_data[idx] == '1':
            quotient += 1
            idx += 1
        idx += 1  # Skip the '0'

        # Decode remainder part
        remainder = int(encoded_data[idx:idx + 8], 2)
        idx += 8

        value = quotient * m + remainder
        decoded_data.append(value)

    return bytes(decoded_data)

# Load the image, compress, and decompress using Golomb coding
def golomb_image_compression(image_path, m):
    img = Image.open(image_path).convert("L")  # Grayscale
    img_data = np.array(img).flatten()

    # Compress
    encoded_data = golomb_encode(img_data, m)
    compressed_size = len(encoded_data) / 8  # Size in bytes
    compression_ratio = compressed_size / len(img_data)

    # Decompress
    decompressed_data = golomb_decode(encoded_data, m)
    decompressed_image = np.array(list(decompressed_data), dtype=np.uint8).reshape(img.size[::-1])

    return img, decompressed_image, len(img_data), compressed_size

# Main function
image_path = "input_image.jpg"  # Replace with your image path
m = 256  # Divisor for Golomb coding
original_img, decompressed_img, original_size, compressed_size = golomb_image_compression(image_path, m)

original_size_mb = original_size / (1024 * 1024)  # Convert to MB
compressed_size_mb = compressed_size / (1024 * 1024)  # Convert to MB

print(f"Original Image Size: {original_size_mb:.2f} MB")
print(f"Compressed Image Size: {compressed_size_mb:.2f} MB")

# Show results
plt.figure(figsize=(12, 6))

# Original image
plt.subplot(1, 2, 1)
plt.title(f"Original Image\nSize: {original_size_mb:.2f} MB")
plt.imshow(original_img, cmap="gray")

# Decompressed image
plt.subplot(1, 2, 2)
plt.title(f"Decompressed Image\nSize: {compressed_size_mb:.2f} MB")
plt.imshow(decompressed_img, cmap="gray")

plt.show()
