#!/usr/bin/env python
# coding: utf-8

# # FlowNet
# A deep learning approach that uses a convolutional neural network to learn optical flow estimation from synthetic training data. FlowNet is designed to predict optical flow in a single forward pass, making it faster than traditional iterative methods but still moderate in terms of speed and computation.

# ## Importing necessary libraries and notebooks

# In[ ]:


import xarray as xr
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

# Append the parent directory (Sargassum) to the system path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
# Import the other notebooks without running their cells
from ii_Data_Manipulation import visualize_4
from iii_GOES_average import time_list, visualize_aggregate, calculate_median
from iv_Image_Processing import collect_times, crop_image, save_aggregate, binarize_image, bilateral_image, process_dates, process_directory
from v_i_OF_Functions import *


# In[ ]:


import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os0

sys.path.append('flownet2-pytorch')

from models import FlowNet2  # the path is depended on where you create this module
from utils.frame_utils import read_gen  # the path is depended on where you create this module
from utils.flow_utils import flow2img

# Load FlowNet2 model
def load_flownet2():
    model = FlowNet2()
    checkpoint = torch.load('FlowNet2_checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.eval()
    return model

# Preprocess images
def preprocess_image(image_path):
    img = read_gen(image_path)
    img = img.astype(np.float32).transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).cuda()
    return img

# Compute optical flow
def compute_flow(model, img1, img2):
    with torch.no_grad():
        result = model(img1, img2)
    flow = result[0].cpu().data.numpy().transpose(1, 2, 0)
    return flow

# Plot flow vectors
def plot_flow_vectors(flow, img, step=16, scale=1.25):
    h, w = flow.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].astype(int)
    fx, fy = flow[y, x].T

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.quiver(x, y, fx, fy, color='r', scale=scale, width=0.002, headwidth=3, headlength=4)
    plt.show()

# Main function
if __name__ == "__main__":
    model = load_flownet2()

    img1_path = "/path/to/first/image.png"
    img2_path = "/path/to/second/image.png"

    img1 = preprocess_image(img1_path)
    img2 = preprocess_image(img2_path)

    flow = compute_flow(model, img1, img2)

    img1_display = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
    plot_flow_vectors(flow, img1_display)

