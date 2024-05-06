import cv2
import numpy as np
import os

# Specify the directory containing images
input_dir = './Test_studentversion/images'
output_dir = './Train/enhanced_images'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over each file in the directory
for filename in os.listdir(input_dir):
    # Construct the full file path
    file_path = os.path.join(input_dir, filename)
    # Check if the file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Load the image in color
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        # Apply bilateral filter to each color channel
        filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        # Convert to grayscale for edge enhancement
        gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        # Create a 3 channel version of the laplacian
        laplacian_3channel = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        # Combine the filtered image with the enhanced edges
        sharp_image = cv2.addWeighted(filtered_image, 1, laplacian_3channel, -1, 0)
        # Save the enhanced image
        output_file_path = os.path.join(output_dir, 'enhanced_' + filename)
        cv2.imwrite(output_file_path, sharp_image)

print("Processing complete.")
