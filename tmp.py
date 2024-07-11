import os
from PIL import Image

import os

def center_crop(image, target_width, target_height):
    width, height = image.size
    left = (width - target_width) / 2
    top = (height - target_height) / 2
    right = (width + target_width) / 2
    bottom = (height + target_height) / 2
    return image.crop((left, top, right, bottom))

def process_images(folder_path):
    target_size = (512, 512)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            with Image.open(file_path) as img:
                if img.size != target_size:
                    cropped_img = center_crop(img, *target_size)
                    cropped_img.save(file_path)

# Define your input and output folders
input_folder = '/home/hngan/BrushNet/BrushNet/examples/brushnet/validation/images'

# Process images
process_images(input_folder)
