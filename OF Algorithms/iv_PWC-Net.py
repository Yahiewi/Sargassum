#!/usr/bin/env python
# coding: utf-8

# # PWC-Net
# Utilizes a pyramid processing framework, warping, and cost volume techniques. It is designed for both accuracy and efficiency, making it suitable for real-time applications that require high-quality flow estimation.

# ## Importing necessary libraries and notebooks

# In[ ]:


import torch
import subprocess
import io
import os
import cv2
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from matplotlib import ticker
from IPython.display import Image, display
from PIL import Image as PILImage

# Append the cloned repository to the system path
sys.path.append(os.path.abspath('/home/yahia/Documents/GitProjects/pytorch-pwc'))
# Import the necessary functions from pwc_net
# from pwc_net import estimate

# Append the parent directory (Sargassum) to the system path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
# Import the other notebooks without running their cells
from ii_Data_Manipulation import visualize_4
from iii_GOES_average import time_list, visualize_aggregate, calculate_median
from iv_Image_Processing import collect_times, crop_image, save_aggregate, binarize_image, bilateral_image, process_dates, process_directory
from v_i_OF_Functions import *


# ## PWC-Net Algorithm

# In[ ]:


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img


# In[ ]:


# Path to the modified run.py
run_py_path = '/home/yahia/Documents/GitProjects/pytorch-pwc/run.py'

# Sample image paths
image_one_path = '/home/yahia/Documents/GitProjects/pytorch-pwc/images/one.png'
image_two_path = '/home/yahia/Documents/GitProjects/pytorch-pwc/images/two.png'

# Ensure paths are absolute if needed
run_py_path = os.path.abspath(run_py_path)
image_one_path = os.path.abspath(image_one_path)
image_two_path = os.path.abspath(image_two_path)

# Output file path
output_path = os.path.abspath('/home/yahia/Documents/GitProjects/pytorch-pwc/images/out.flo')

# Run the modified script
subprocess.run(['python', run_py_path, '--model', 'default', '--one', image_one_path, '--two', image_two_path, '--out', output_path])

# Function to read .flo files
def read_flo_file(file_path):
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert magic == 202021.25, "Magic number incorrect. Invalid .flo file"
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
        data2D = np.resize(data, (h, w, 2))
        return data2D

# Read and visualize the output
flow = read_flo_file(output_path)
plt.imshow(flow[:, :, 0], cmap='gray')
plt.title('Optical Flow (x component)')
plt.show()

plt.imshow(flow[:, :, 1], cmap='gray')
plt.title('Optical Flow (y component)')
plt.show()


# In[ ]:


# Load images
prev_img_path = "/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral_Negative/Processed_algae_distribution_20220723.png"
next_img_path = "/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral_Negative/Processed_algae_distribution_20220724.png"
prev_img = preprocess_image(prev_img_path)
next_img = preprocess_image(next_img_path)

# Compute optical flow
device = 'cpu'  # or 'cuda' if you have a GPU
flow = estimate(prev_img, next_img, model_type='default', device=device)

# Convert the flow to a numpy array
flow = flow.numpy().transpose(1, 2, 0)

# Plot flow vectors on the base image
flow_plot = plot_flow_vectors(flow, prev_img_path, step=16, scale=1, display=True, color='r')


# In[ ]:




