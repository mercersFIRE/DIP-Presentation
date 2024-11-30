import numpy as np
import pywt
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

# Perform Wavelet Transform and compression
def wavelet_coding(image_path, wavelet='haar'):
    img = Image.open(image_path).convert("L")  # Convert image to grayscale
    img_data = np.array(img)
    
    # Apply 2D Discrete Wavelet Transform (DWT)
    coeffs2 = pywt.dwt2(img_data, wavelet)
    LL, (LH, HL, HH) = coeffs2  # Extract coefficients (LL = Low-Low, LH = Low-High, etc.)
    
    # For simplicity, we will compress all coefficients using RLE
    LL_flat = LL.flatten()
    LH_flat = LH.flatten()
    HL_flat = HL.flatten()
    HH_flat = HH.flatten()
    
    # Compress each coefficient array using RLE
    compressed_LL = rle_compress(LL_flat)
    compressed_LH = rle_compress(LH_flat)
    compressed_HL = rle_compress(HL_flat)
    compressed_HH = rle_compress(HH_flat)
    
    # Decompress the coefficients
    decompressed_LL = rle_decompress(compressed_LL).reshape(LL.shape)
    decompressed_LH = rle_decompress(compressed_LH).reshape(LH.shape)
    decompressed_HL = rle_decompress(compressed_HL).reshape(HL.shape)
    decompressed_HH = rle_decompress(compressed_HH).reshape(HH.shape)
    
    # Reconstruct the image using Inverse DWT
    decompressed_coeffs2 = decompressed_LL, (decompressed_LH, decompressed_HL, decompressed_HH)
    decompressed_img_data = pywt.idwt2(decompressed_coeffs2, wavelet)
    
    decompressed_image = Image.fromarray(np.uint8(decompressed_img_data))
    
    return img, decompressed_image, img.size, decompressed_image.size

# Main function
image_path = "input_image.jpg"  # Replace with your image path
original_img, decompressed_img, original_size, decompressed_size = wavelet_coding(image_path)

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
