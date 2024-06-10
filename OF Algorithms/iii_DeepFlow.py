#!/usr/bin/env python
# coding: utf-8

# # DeepFlow
# A method that combines a traditional variational approach with deep learning-based matching (using a CNN). It's particularly effective at capturing large displacements, which makes it robust in complex, real-world scenes.

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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Append the parent directory (Sargassum) to the system path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
# Import the other notebooks without running their cells
from ii_Data_Manipulation import visualize_4
from iii_GOES_average import time_list, visualize_aggregate, calculate_median
from iv_Image_Processing import collect_times, crop_image, save_aggregate, binarize_image, bilateral_image, process_dates, process_directory
from v_i_OF_Functions import *


# ## DeepFlow Algorithm

# In[ ]:


def deepflow(prev_img, next_img):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    
    # Initialize DeepFlow
    deep_flow = cv2.optflow.createOptFlow_DeepFlow()
    
    # Compute flow
    flow = deep_flow.calc(prev_gray, next_gray, None)
    
    return flow


# ### 23/07 - 24/07

# In[ ]:


if __name__ == '__main__':
    # Binarized (negative)
    prev_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral_Negative/Processed_algae_distribution_20220723.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral_Negative/Processed_algae_distribution_20220724.png")
    flow = deepflow(prev_img, next_img)
    plot_flow_vectors(flow, prev_img, step=16, scale=1.25)
    # GIF
    # THIS SHOULD BE FIXED: The GIF function we wrote gives an inverted GIF so we have to invert next_img and prev_img and then reverse the interpolated_images list.
    interpolated_images = interpolate_images(next_img, prev_img, flow, num_interpolations=60)
    interpolated_images.reverse()
    visualize_movement(interpolated_images, fps=30)
    
    # Viridis
    prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220723.png")
    next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220724.png")
    flow = deepflow(prev_img, next_img)
    plot_flow_vectors(flow, prev_img, step=16, scale=1.25)
    # GIF
    # THIS SHOULD BE FIXED: The GIF function we wrote gives an inverted GIF so we have to invert next_img and prev_img and then reverse the interpolated_images list.
    interpolated_images = interpolate_images(next_img, prev_img, flow, num_interpolations=60)
    interpolated_images.reverse()
    visualize_movement(interpolated_images, fps=30)


# ### 05/18 - 05/19

# In[ ]:


if __name__ == '__main__':
    # Binarized (negative)
    prev_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral_Negative/Processed_algae_distribution_20220518.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral_Negative/Processed_algae_distribution_20220519.png")
    flow = deepflow(prev_img, next_img)
    plot_flow_vectors(flow, prev_img, step=16, scale=1.25)
    # GIF
    # THIS SHOULD BE FIXED: The GIF function we wrote gives an inverted GIF so we have to invert next_img and prev_img and then reverse the interpolated_images list.
    interpolated_images = interpolate_images(next_img, prev_img, flow, num_interpolations=60)
    interpolated_images.reverse()
    visualize_movement(interpolated_images, fps=30)
    
    # Viridis
    prev_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Processed/Processed_algae_distribution_20220518.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Processed/Processed_algae_distribution_20220519.png")
    flow = deepflow(prev_img, next_img)
    plot_flow_vectors(flow, prev_img, step=16, scale=1.25)
    # GIF
    # THIS SHOULD BE FIXED: The GIF function we wrote gives an inverted GIF so we have to invert next_img and prev_img and then reverse the interpolated_images list.
    interpolated_images = interpolate_images(next_img, prev_img, flow, num_interpolations=60)
    interpolated_images.reverse()
    visualize_movement(interpolated_images, fps=30)


# ### 05/21 - 05/22

# In[ ]:


if __name__ == '__main__':
    # Binarized (negative)
    prev_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral_Negative/Processed_algae_distribution_20220521.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral_Negative/Processed_algae_distribution_20220522.png")
    flow = deepflow(prev_img, next_img)
    plot_flow_vectors(flow, prev_img, step=16, scale=1.25)
    # Viridis
    prev_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Processed/Processed_algae_distribution_20220521.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Processed/Processed_algae_distribution_20220522.png")
    flow = deepflow(prev_img, next_img)
    plot_flow_vectors(flow, prev_img, step=16, scale=1.25)
    # GIF
    # THIS SHOULD BE FIXED: The GIF function we wrote gives an inverted GIF so we have to invert next_img and prev_img and then reverse the interpolated_images list.
    interpolated_images = interpolate_images(next_img, prev_img, flow, num_interpolations=60)
    interpolated_images.reverse()
    visualize_movement(interpolated_images, fps=30)


# ### 07/03 - 07/04

# In[ ]:


if __name__ == '__main__':
    # Binarized (negative)
    prev_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral_Negative/Processed_algae_distribution_20220703.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral_Negative/Processed_algae_distribution_20220704.png")
    flow = deepflow(prev_img, next_img)
    plot_flow_vectors(flow, prev_img, step=16, scale=1.25)
    # GIF
    # THIS SHOULD BE FIXED: The GIF function we wrote gives an inverted GIF so we have to invert next_img and prev_img and then reverse the interpolated_images list.
    interpolated_images = interpolate_images(next_img, prev_img, flow, num_interpolations=60)
    interpolated_images.reverse()
    visualize_movement(interpolated_images, fps=30)
    # Viridis
    prev_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Processed/Processed_algae_distribution_20220703.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Processed/Processed_algae_distribution_20220704.png")
    flow = deepflow(prev_img, next_img)
    plot_flow_vectors(flow, prev_img, step=16, scale=1.25)
    # GIF
    # THIS SHOULD BE FIXED: The GIF function we wrote gives an inverted GIF so we have to invert next_img and prev_img and then reverse the interpolated_images list.
    interpolated_images = interpolate_images(next_img, prev_img, flow, num_interpolations=60)
    interpolated_images.reverse()
    visualize_movement(interpolated_images, fps=30)


# ### 07/13 - 07/14

# In[ ]:


if __name__ == '__main__':
    # Binarized (negative)
    prev_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral_Negative/Processed_algae_distribution_20220713.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral_Negative/Processed_algae_distribution_20220714.png")
    flow = deepflow(prev_img, next_img)
    plot_flow_vectors(flow, prev_img, step=16, scale=1.25)
    # Viridis
    prev_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Processed/Processed_algae_distribution_20220713.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Processed/Processed_algae_distribution_20220714.png")
    flow = deepflow(prev_img, next_img)
    plot_flow_vectors(flow, prev_img, step=16, scale=1.25)


# ## Zoom

# In[ ]:


if __name__ == '__main__':
    # Viridis
    prev_img = cv2.imread("/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral_Processed/Processed_algae_distribution_20220723.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral_Processed/Processed_algae_distribution_20220724.png")
    flow = deepflow(prev_img, next_img)
    plot_flow_vectors(flow, prev_img, step=16, scale=1.25)
    # GIF
    # THIS SHOULD BE FIXED: The GIF function we wrote gives an inverted GIF so we have to invert next_img and prev_img and then reverse the interpolated_images list.
    interpolated_images = interpolate_images(next_img, prev_img, flow, num_interpolations=60)
    interpolated_images.reverse()
    visualize_movement(interpolated_images, fps=30)
    # Binarized
    prev_img = cv2.imread("/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral_Binarized_Bilateral/Processed_algae_distribution_20220723.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral_Binarized_Bilateral/Processed_algae_distribution_20220724.png")
    flow = deepflow(prev_img, next_img)
    plot_flow_vectors(flow, prev_img, step=16, scale=1.25)
    # GIF
    # THIS SHOULD BE FIXED: The GIF function we wrote gives an inverted GIF so we have to invert next_img and prev_img and then reverse the interpolated_images list.
    interpolated_images = interpolate_images(next_img, prev_img, flow, num_interpolations=60)
    interpolated_images.reverse()
    visualize_movement(interpolated_images, fps=30)

