import os
from PIL import Image
from PIL import Image, UnidentifiedImageError

imageFolder = "/lustre/scratch/client/vinai/users/ngannh9/oldhand/data/LAION/preprocessed_2256k/train"
with open("/lustre/scratch/client/vinai/users/ngannh9/oldhand/LAVIS/human.txt", 'r') as file:
    lines = file.readlines()
listImages = [os.path.join(imageFolder, line.strip()) for line in lines]
valid_images = []
corrupted_images = []
for img in listImages:
    imgPath = os.path.join(imageFolder,img)
    try:
        with Image.open(imgPath) as img:
            img.verify()  # Verify if the image can be opened
            valid_images.append(imgPath)
    except (UnidentifiedImageError, IOError, SyntaxError) as e:
        corrupted_images.append(imgPath)
        print(f"Corrupted image found: {imgPath} - {e}")
