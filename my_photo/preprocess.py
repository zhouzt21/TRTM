import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from PIL import Image
import numpy as np
import cv2
import os

def read_depth_image(file_path):
    img = mpimg.imread(file_path)
    img_gray = rgb2gray(img)
    return img_gray

def upscale_image(img, new_width, new_height):
    return cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR)

def crop_image(img, left, upper, right, lower):
    img_pil = Image.fromarray(img)
    img_cropped = img_pil.crop((left, upper, right, lower))
    return np.array(img_cropped)

def process_images_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('_Depth.png'):
            file_path = os.path.join(directory_path, filename)
            img = read_depth_image(file_path)
            img_upscaled = upscale_image(img, 848,480)
            img_cropped = crop_image(img_upscaled, 200, 0, 680, 480)
            img_processed = upscale_image(img_cropped, 720,720)
            plt.imsave(filename.replace('_Depth.png', '.real_depth.png'), img_processed, cmap='gray')

# process_images_in_directory('/home/rob/TRTM/my_photo/origin')

# img = read_depth_image('./origin/shirt_sofa_Depth.png')
# img_upscaled = upscale_image(img, 848,480)
# img_cropped = crop_image(img_upscaled, 100, 0, 580, 480)
# img_processed = upscale_image(img_cropped, 720,720)

# plt.imshow(img_cropped, cmap='gray')
# plt.colorbar(label='Depth')
# plt.show()
# plt.imsave('shirt_sofa.real_depth.png', img_cropped, cmap='gray')

########################-------------#####################

def read_color_image(file_path):
    img = mpimg.imread(file_path)
    return img

def crop_color_image(img, left, upper, right, lower):
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    img_cropped = img_pil.crop((left, upper, right, lower))
    return np.array(img_cropped) / 255.0

def process_color_images_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('_Color.png'):
            file_path = os.path.join(directory_path, filename)
            img = read_color_image(file_path)
            img_scaled = upscale_image(img, 848,480)
            img_cropped = crop_color_image(img_scaled, 160, 0, 640, 480)
            img_processed = upscale_image(img_cropped, 720,720)
            plt.imsave(filename.replace('_Color.png', '.real_color.png'), img_processed)

# process_color_images_in_directory('/home/rob/TRTM/my_photo/origin')

# img = read_color_image('./origin/coat_sofa_Color.png')
# # print(img.shape)
# img_scaled = upscale_image(img, 848,480)
# img_cropped = crop_color_image(img_scaled, 200, 0, 680, 480)
# img_processed = upscale_image(img_cropped, 720,720)


# plt.imsave( 'coat_sofa.real_color.png', img_processed)
