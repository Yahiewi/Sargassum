#!/usr/bin/env python
# coding: utf-8

# # Lucas-Kanade

# Similarly to the previous notebook, in this one we're going to test out the Lucas-Kanade method with different parameters each time.
# 
# This method estimates the motion of objects between consecutive image frames by tracking the displacement of specific feature points in a local neighborhood, assuming constant motion within that neighborhood.

# ## Importing necessary libraries and notebooks

# In[1]:


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

# Import the other notebooks without running their cells
from ii_Data_Manipulation import visualize_4
from iii_GOES_average import time_list, visualize_aggregate, calculate_median
from iv_Image_Processing import collect_times, crop_image, save_aggregate, binarize_image, bilateral_image, process_dates, process_directory
from v_i_OF_Functions import *


# ## Lucas-Kanade
# Like the notebook Farneback, even though this function is already defined in *v_i_OF_Functions* we redefine it here for convenience (so we can change it without having to change it in the other notebook, then reimporting).

# In[2]:


def LK(prev_img, next_img, max_corners=100, quality_level=0.3, min_distance=7, block_size=7, win_size=(15, 15), max_level=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)):
    """
    Computes optical flow using the Lucas-Kanade method.

    :param prev_img: The previous image frame.
    :param next_img: The next image frame.
    :param max_corners: Maximum number of corners to detect.
    :param quality_level: Quality level for corner detection.
    :param min_distance: Minimum possible Euclidean distance between the returned corners.
    :param block_size: Size of an average block for computing a derivative covariance matrix over each pixel neighborhood.
    :param win_size: Size of the search window at each pyramid level.
    :param max_level: 0-based maximal pyramid level number.
    :param criteria: Criteria for termination of the iterative search algorithm.
    :return: p0, p1, st, err
        p0: Initial points in the previous image.
        p1: Corresponding points in the next image.
        st: Status array indicating whether the flow for the corresponding feature has been found.
        err: Error for each point.
    """
    # Ensure images are grayscale
    if len(prev_img.shape) == 3:
        prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    if len(next_img.shape) == 3:
        next_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    
    # Detect good features to track in the previous image
    p0 = cv2.goodFeaturesToTrack(prev_img, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance, blockSize=block_size)
    
    # Calculate optical flow between the two images
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_img, next_img, p0, None, winSize=win_size, maxLevel=max_level, criteria=criteria)
    
    return p0, p1, st, err


# ### 23/07 - 24/07

# In[5]:


if __name__ == '__main__':
    prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220723.png")
    next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png")
    # Compute optical flow using Lucas-Kanade method
    p0, p1, st, err = LK(prev_img, next_img)
    
    # Plot flow vectors on the base image
    flow_plot = plot_LK_vectors(p0, p1, st, prev_img, display=True, color='r')
    #display_image_mpl(flow_plot)

