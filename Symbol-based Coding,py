import heapq
from collections import defaultdict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Build Huffman Tree
def build_huffman_tree(data):
    freq = defaultdict(int)
    for byte in data:
        freq[byte] += 1

    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        low = heapq.heappop(heap)
        high = heapq.heappop(heap)
        for pair in low[1:]:
            pair[1] = '0' + pair[1]
        for pair in high[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [low[0] + high[0]] + low[1:] + high[1:])
    
    return dict(sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p)))

# Encode the image data
def huffman_encode(data, huffman_table):
    return "".join(huffman_table[byte] for byte in data)

# Decode the image data
def huffman_decode(encoded_data, huffman_table):
    reverse_table = {v: k for k, v in huffman_table.items()}
    current_code = ""
    decoded_data = []
    
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_table:
            decoded_data.append(reverse_table[current_code])
            current_code = ""
    return bytes(decoded_data)

# Compress and Decompress Image
def symbol_based_compression(image_path):
    img = Image.open(image_path).convert("L")  # Grayscale
    img_data = np.array(img).flatten()
    
    # Build Huffman Table
    huffman_table = build_huffman_tree(img_data)
    
    # Compress
    encoded_data = huffman_encode(img_data, huffman_table)
    compressed_size = len(encoded_data) / 8  # Size in bytes
    compression_ratio = compressed_size / len(img_data)
    
    # Decompress
    decompressed_data = huffman_decode(encoded_data, huffman_table)
    decompressed_image = np.array(list(decompressed_data), dtype=np.uint8).reshape(img.size[::-1])
    
    return img, decompressed_image, len(img_data), compressed_size

# Main function
image_path = "input_image.jpg"  # Replace with your image path
original_img, decompressed_img, original_size, compressed_size = symbol_based_compression(image_path)

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
