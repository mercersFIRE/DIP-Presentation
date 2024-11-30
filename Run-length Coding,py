from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Run-Length Encoding Compression
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

# Run-Length Encoding Decompression
def rle_decompress(compressed_data):
    decompressed_data = []
    
    for symbol, count in compressed_data:
        decompressed_data.extend([symbol] * count)
    
    return decompressed_data

# Compress and Decompress Image
def rle_image_compression(image_path):
    img = Image.open(image_path).convert("L")  # Grayscale
    img_data = np.array(img).flatten()
    
    # Compress
    compressed_data = rle_compress(img_data)
    compressed_size = len(compressed_data) * 2  # Approx size in bytes (2 bytes per (value, count) pair)
    compression_ratio = compressed_size / len(img_data)
    
    # Decompress
    decompressed_data = rle_decompress(compressed_data)
    decompressed_image = np.array(decompressed_data, dtype=np.uint8).reshape(img.size[::-1])
    
    return img, decompressed_image, len(img_data), compressed_size

# Main function
image_path = "input_image.jpg"  # Replace with your image path
original_img, decompressed_img, original_size, compressed_size = rle_image_compression(image_path)

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
