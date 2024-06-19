import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

root_dir = 'C:\python\projects\pythonProject\DeepGlobe_dataset\DeepGlobe dataset\\train'
mask_dir = 'C:\python\projects\pythonProject\DeepGlobe_dataset\DeepGlobe dataset\modified_masks'
files = os.listdir(root_dir)
png_files = []
png_files_full = []
for file in files:
    if (file.endswith('.png')):
        png_files.append(file)

for pngfile in png_files:
    png_path = os.path.join(root_dir,  pngfile)
    png_files_full.append(png_path)

for reading in png_files_full:
    img = cv.imread(reading)
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    hsv_color1 = np.asarray([0, 1, 255])  # white!
    hsv_color2 = np.asarray([30, 255, 255])  # yellow!

    mask = cv.inRange(img_hsv, hsv_color1, hsv_color2)
    image_filename = os.path.splitext(os.path.basename(reading))[0]  # Get filename without extension
    mask_filename = f"{image_filename}_mask.png"
    mask_file_full_dir = os.path.join(mask_dir, mask_filename)
    cv.imwrite(mask_file_full_dir, mask)

    print(f"Mask saved to: {mask_file_full_dir}")

#deleting the mask files from training folder

import os, glob

for file in glob.glob("C:\python\projects\pythonProject\DeepGlobe_dataset\DeepGlobe dataset\\train\\*_mask*"):
    print(os.remove(file))


