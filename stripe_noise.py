import cv2
import numpy as np
import os
from tqdm import tqdm


def replace_specific_color_range(image, color_ranges):
    # Convert image to HSV for better color filtering
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    for lower_bound, upper_bound in color_ranges:
        # Create a mask for the color range in HSV
        mask = cv2.inRange(hsv_image, np.array(lower_bound), np.array(upper_bound))
        
        # Find indices where mask is true
        indices = np.where(mask > 0)
        
        # Replace color with the average of left and right pixels
        for y, x in zip(indices[0], indices[1]):
            if x > 0 and x < image.shape[1] - 1:
                left_pixel = image[y, max(x - 4, 0)]
                image[y, x] = left_pixel
                image[y, max(x-1,0)] = left_pixel
                image[y, min(x+1,1023)] = left_pixel
                image[y, max(x-2,0)] = left_pixel
    
    return image

def process_image(file_path, output_path):
    # Load image
    image = cv2.imread(file_path)
    if image is None:
        print("Error loading image")
        return

    # Convert image to RGB (OpenCV loads images in BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the HSV ranges for green and blue colors
    color_ranges = [
        ([110, 50, 50], [130, 255, 255]),  # Blue color range in HSV
        ([0, 50, 70], [10, 255, 255]),  # Red color range in HSV
        ([40, 60, 100], [80, 255, 255]) # Green color range in HSV
    ]

    # Replace specific color ranges
    cleaned_image = replace_specific_color_range(image, color_ranges)
    # Save the processed image
    cv2.imwrite(output_path, cv2.cvtColor(cleaned_image, cv2.COLOR_RGB2BGR))

def main():

    input_dir = './Test_studentversion/images'
    output_dir = './Test_studentversion/stripe_cleaned_images/'
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all files in the input directory
    file_names = os.listdir(input_dir)

    # Initialize progress bar
    pbar = tqdm(total=len(file_names))

    for file_name in file_names:
        # Skip non-image files
        if not (file_name.endswith('.png') or file_name.endswith('.jpg')):
            continue

        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        process_image(input_path, output_path)

        # Update progress bar
        pbar.update(1)

    # Close progress bar
    pbar.close()

if __name__ == '__main__':
    main()