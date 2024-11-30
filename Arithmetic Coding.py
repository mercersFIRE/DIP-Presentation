from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Arithmetic Coding Implementation
class ArithmeticCoding:
    def __init__(self):
        self.low = 0.0
        self.high = 1.0

    def compress(self, data, probabilities):
        for symbol in data:
            range_ = self.high - self.low
            self.high = self.low + range_ * probabilities[symbol][1]
            self.low = self.low + range_ * probabilities[symbol][0]
        return self.low

    def decompress(self, value, data_len, probabilities):
        result = []
        for _ in range(data_len):
            for symbol, (low, high) in probabilities.items():
                if low <= value < high:
                    result.append(symbol)
                    value = (value - low) / (high - low)
                    break
        return result

# Calculate probabilities of symbols
def calculate_probabilities(data):
    freq = defaultdict(int)
    for symbol in data:
        freq[symbol] += 1
    total = sum(freq.values())
    probabilities = {}
    low = 0.0
    for symbol, count in freq.items():
        high = low + count / total
        probabilities[symbol] = (low, high)
        low = high
    return probabilities

# Compress the image
def arithmetic_image_compression(image_path):
    img = Image.open(image_path).convert("L")  # Grayscale
    img_data = np.array(img).flatten()
    
    # Calculate probabilities and compress
    probabilities = calculate_probabilities(img_data)
    coder = ArithmeticCoding()
    compressed_value = coder.compress(img_data, probabilities)
    
    compressed_size = len(img_data) * 32 / 8  # Floating-point number as compressed size in bytes
    compression_ratio = compressed_size / len(img_data)
    
    # Decompress
    decompressed_data = coder.decompress(compressed_value, len(img_data), probabilities)
    decompressed_image = np.array(decompressed_data, dtype=np.uint8).reshape(img.size[::-1])
    
    return img, decompressed_image, len(img_data), compressed_size

# Main function
image_path = "input_image.jpg"  # Replace with your image path
original_img, decompressed_img, original_size, compressed_size = arithmetic_image_compression(image_path)

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
