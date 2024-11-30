from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Function to extract the bit-planes of an image
def extract_bit_planes(image_data):
    bit_planes = []
    for i in range(8):  # For 8-bit image
        bit_plane = (image_data >> i) & 1  # Extract each bit-plane
        bit_planes.append(bit_plane)
    return bit_planes

# Run-Length Encoding (RLE) for compression
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

# Function to compress and decompress the bit-planes of an image
def bit_plane_coding(image_path):
    img = Image.open(image_path).convert("L")  # Grayscale image
    img_data = np.array(img).flatten()

    # Extract bit-planes
    bit_planes = extract_bit_planes(img_data)
    
    # Compress each bit-plane using RLE
    compressed_bit_planes = []
    for plane in bit_planes:
        compressed_plane = rle_compress(plane)
        compressed_bit_planes.append(compressed_plane)

    # Decompress each bit-plane using RLE
    decompressed_bit_planes = []
    for compressed_plane in compressed_bit_planes:
        decompressed_plane = rle_decompress(compressed_plane)
        decompressed_bit_planes.append(decompressed_plane)

    # Reconstruct the image from decompressed bit-planes
    decompressed_image_data = np.zeros_like(img_data)
    for i, plane in enumerate(decompressed_bit_planes):
        decompressed_image_data |= (plane << i)

    decompressed_image = decompressed_image_data.reshape(img.size[::-1])

    return img, decompressed_image, len(img_data), sum(len(plane) * 2 for plane in compressed_bit_planes) / 8  # Approx size in bytes

# Main function
image_path = "input_image.jpg"  # Replace with your image path
original_img, decompressed_img, original_size, compressed_size = bit_plane_coding(image_path)

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
