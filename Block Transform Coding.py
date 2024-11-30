import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.fftpack as fft

# Function to perform DCT on an 8x8 block
def dct2(block):
    return fft.dct(fft.dct(block.T, norm='ortho').T, norm='ortho')

# Function to perform Inverse DCT on an 8x8 block
def idct2(block):
    return fft.idct(fft.idct(block.T, norm='ortho').T, norm='ortho')

# Quantization matrix for 8x8 blocks (example matrix, you can adjust)
quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Function to apply quantization
def quantize(dct_block, quantization_matrix):
    return np.round(dct_block / quantization_matrix)

# Function to compress and decompress the image
def block_transform_coding(image_path):
    img = Image.open(image_path).convert("L")  # Convert image to grayscale
    img_data = np.array(img)
    
    # Define block size (8x8)
    block_size = 8
    
    # Get the dimensions of the image
    height, width = img_data.shape
    
    # Make sure the image size is a multiple of block_size
    new_height = height - (height % block_size)
    new_width = width - (width % block_size)
    img_data = img_data[:new_height, :new_width]
    
    # Split the image into 8x8 blocks
    blocks = [
        img_data[i:i+block_size, j:j+block_size]
        for i in range(0, new_height, block_size)
        for j in range(0, new_width, block_size)
    ]
    
    # Apply DCT to each block and quantize
    compressed_blocks = []
    for block in blocks:
        dct_block = dct2(block)
        quantized_block = quantize(dct_block, quantization_matrix)
        compressed_blocks.append(quantized_block)
    
    # Reconstruct the image by applying Inverse DCT
    decompressed_image_data = np.zeros_like(img_data)
    idx = 0
    for i in range(0, new_height, block_size):
        for j in range(0, new_width, block_size):
            decompressed_block = idct2(compressed_blocks[idx])
            decompressed_image_data[i:i+block_size, j:j+block_size] = np.round(decompressed_block)
            idx += 1
    
    # Resize back to original dimensions (if necessary)
    decompressed_image = Image.fromarray(decompressed_image_data.astype(np.uint8))
    
    return img, decompressed_image, img.size, decompressed_image.size

# Main function
image_path = "input_image.jpg"  # Replace with your image path
original_img, decompressed_img, original_size, decompressed_size = block_transform_coding(image_path)

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
