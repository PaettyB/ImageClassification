from PIL import Image
import os
from pathlib import Path

original_size = 640
target_size = 400

crop = (original_size - target_size) / 2

pathlist = Path("archive").glob("**/*.jpg")
for path in pathlist:
    tokens = str(path).split("\\")
    img = Image.open(path)
    crp = img.crop((crop, crop, original_size - crop, original_size - crop))
    crp.save(f"archive_crop/{tokens[1]}/{tokens[2]}/{tokens[3]}")

# crp.save("archive_crop/train/turtle/image_id_002.jpg")