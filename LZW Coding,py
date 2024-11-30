from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# LZW Compression
def lzw_compress(data):
    dictionary = {bytes([i]): i for i in range(256)}
    current_string = b""
    compressed_data = []
    next_code = 256
    
    for symbol in data:
        current_symbol = bytes([symbol])
        if current_string + current_symbol in dictionary:
            current_string += current_symbol
        else:
            compressed_data.append(dictionary[current_string])
            if next_code < 4096:  # Limit dictionary size
                dictionary[current_string + current_symbol] = next_code
                next_code += 1
            current_string = current_symbol
    
    if current_string:
        compressed_data.append(dictionary[current_string])
    
    return compressed_data

# LZW Decompression
def lzw_decompress(compressed_data):
    dictionary = {i: bytes([i]) for i in range(256)}
    next_code = 256
    current_string = bytes([compressed_data.pop(0)])
    decompressed_data = bytearray(current_string)
    
    for code in compressed_data:
        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code:
            entry = current_string + current_string[:1]
        else:
            raise ValueError("Invalid LZW compressed data")
        
        decompressed_data.extend(entry)
        if next_code < 4096:  # Limit dictionary size
            dictionary[next_code] = current_string + entry[:1]
            next_code += 1
        current_string = entry
    
    return decompressed_data

# Compress and Decompress Image
def lzw_image_compression(image_path):
    img = Image.open(image_path).convert("L")  # Grayscale
    img_data = np.array(img).flatten()
    
    # Compress
    compressed_data = lzw_compress(img_data)
    compressed_size = len(compressed_data) * 2  # Approx size in bytes (16 bits per code)
    compression_ratio = compressed_size / len(img_data)
    
    # Decompress
    decompressed_data = lzw_decompress(compressed_data)
    decompressed_image = np.array(decompressed_data, dtype=np.uint8).reshape(img.size[::-1])
    
    return img, decompressed_image, len(img_data), compressed_size

# Main function
image_path = "input_image.jpg"  # Replace with your image path
original_img, decompressed_img, original_size, compressed_size = lzw_image_compression(image_path)

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
