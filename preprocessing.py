import os
from skimage import filters, img_as_ubyte, restoration, io, img_as_float
from PIL import Image
import numpy as np
from tqdm import tqdm

def preprocess_and_denoise_images(root, output_folder='denoised_images'):
    os.makedirs(output_folder, exist_ok=True)
    img_paths = os.listdir(root)
    for img_path in tqdm(img_paths, desc="Processing images"):
        img_path = os.path.join(root, img_path)
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        img_np = img_as_float(img_np)
        # Apply median filtering
        img_np = filters.median(img_np)
        # Apply Gaussian filtering
        img_np = filters.gaussian(img_np, sigma=1, mode='reflect', cval=0, preserve_range=True, truncate=4.0)
        # Apply wavelet denoising
        img_np = restoration.denoise_wavelet(img_np)
        try:
            # Save the denoised images
            img_denoised = img_as_ubyte(img_np)
            filename = os.path.basename(img_path)  # Extract filename from img_path
            io.imsave(os.path.join(output_folder, filename), img_denoised)  # Use the same filename for the denoised image
        except:
            pass

if __name__ == "__main__":
    # Example usage
    preprocess_and_denoise_images(root='./Train/images', output_folder='./Train/denoised_images')