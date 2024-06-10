#!/usr/bin/env python
# coding: utf-8

# # Motion Estimation
# Using the processed images produced by the fourth notebook, we're going to apply Optical Flow algorithms from the OpenCV library to estimate the motion of the algae.

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
from concurrent.futures import ProcessPoolExecutor

# Import the other notebooks without running their cells
from ii_Data_Manipulation import visualize_4
from iii_GOES_average import time_list, visualize_aggregate, calculate_median
from iv_Image_Processing import collect_times, crop_image, save_aggregate, binarize_image, bilateral_image, process_dates, process_directory
from v_i_OF_Functions import *


# ## Optical Flow Implementations

# ### Visualizing the Flow

# In[ ]:


# if __name__ == '__main__':
#     prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220723.png")
#     next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png")
#     flow = farneback_flow(prev_img, next_img)
#     mag, ang = compute_flow_components(flow)
#     visualize_flow_components(mag, ang)


# We can also visualize the motion field through vectors.

# In[ ]:


# if __name__ == '__main__':
#     prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220723.png")
#     next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png")
#     flow = farneback_flow(prev_img, next_img)
#     plot_of_vectors(flow, prev_img, step=16, scale=1.25, display=True)
#     image = overlay_flow_vectors_with_quiver(flow, prev_img)
#     display_image_mpl(image, scale=1)
#     #display(Image(filename="/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png", width =750))


# ### GIF
# We can try to visualize the result using a GIF.

# In[ ]:


# if __name__ == '__main__':
#     # Saving the GIF
#     prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220723.png")
#     next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png")
#     images = [prev_img, next_img]
#     create_flow_gif(images, '/home/yahia/Documents/Jupyter/Sargassum/Images/GIFs/optical_flow.gif', fps=0.2, loop=10)
    
#     # Displaying the GIF
#     gif_path = '/home/yahia/Documents/Jupyter/Sargassum/Images/GIFs/optical_flow.gif' 
#     display(Image(filename=gif_path))


# This algorithm doesn't track the images very well, maybe trying with a viridis color map would produce better results.

# ### Trying different Colormaps

# The binarized image doesn't seem to be adapted for our algorithm, so we'll try to apply our algorithm on Viridis images.

# In[ ]:


# if __name__ == '__main__':
#     prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220723.png")
#     next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220724.png")
#     flow = farneback_flow(prev_img, next_img)
#     plot_of_vectors(flow, prev_img, step=16, scale=1.25)
#     display(Image(filename="/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220724.png", width =750))


# In[ ]:


# if __name__ == '__main__':
#     # Saving the GIF
#     prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220723.png")
#     next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220724.png")
#     images = [prev_img, next_img]
#     create_flow_gif(images, '/home/yahia/Documents/Jupyter/Sargassum/Images/GIFs/optical_flow.gif', fps=0.4, loop=10)
    
#     # Displaying the GIF
#     gif_path = '/home/yahia/Documents/Jupyter/Sargassum/Images/GIFs/optical_flow.gif' 
#     display(Image(filename=gif_path))


# ### Trying Lucas-Kanade

# In[ ]:


# if __name__ == '__main__':
#     # Binarized
#     prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220723.png")
#     next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png")
#     p0, p1, st, err = LK_flow(prev_img, next_img)
#     if p0 is not None and p1 is not None:
#         # Filter out only points with successful tracking
#         good_new = p1[st==1]
#         good_old = p0[st==1]
#         LK1 = LK_vector_field(p0, p1, st[st==1], prev_img)
#         display_image_cv(LK1)
#     # Viridis
#     prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220723.png")
#     next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220724.png")
#     p0, p1, st, err = LK_flow(prev_img, next_img)
#     if p0 is not None and p1 is not None:
#         # Filter out only points with successful tracking
#         good_new = p1[st==1]
#         good_old = p0[st==1]
#         LK2 = LK_vector_field(p0, p1, st[st==1], prev_img)
#         display_image_cv(LK2)


# ### OpenCV Image Display

# In[ ]:


# if __name__ == '__main__':
#     # Binary image
#     prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220723.png")
#     next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png")
#     p0, p1, st, err = LK_flow(prev_img, next_img)
#     img_with_vectors = LK_vector_field(p0, p1, st, prev_img)
#     display_image_cv(img_with_vectors)


# ### Image Superposition
# This is a function that takes two images (preferably binarized for clarity) and superposes them on top of each other with different colors.

# In[ ]:


# if __name__ == '__main__':
#     prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220723.png")
#     next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png")
#     superposed = superpose_images(prev_img, next_img)
#     display_image_cv(superposed)


# In[ ]:


# if __name__ == '__main__':
#     motion_field = overlay_flow_vectors(flow, superposed, step=16, scale=1, color=(0,0,255))
#     display_image_cv(motion_field)


# ## DeepFlow on Atlantic

# In[ ]:


if __name__ == '__main__':
    # Binarized (negative)
    prev_img = cv2.imread("/media/yahia/ballena/ABI/Atlantic/Averages_Binarized_Bilateral/Processed_algae_distribution_20220723.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/Atlantic/Averages_Binarized_Bilateral/Processed_algae_distribution_20220724.png")
    flow = deepflow(prev_img, next_img)
    plot_flow_vectors_opencv(flow, prev_img, step=16, scale=1.25)
    # # Viridis
    # prev_img = cv2.imread("/media/yahia/ballena/ABI/Atlantic/Averages_Cropped/Processed_algae_distribution_20220723.png")
    # next_img = cv2.imread("/media/yahia/ballena/ABI/Atlantic/Averages_Cropped/Processed_algae_distribution_20220724.png")
    # flow = deepflow(prev_img, next_img)
    # plot_flow_vectors(flow, prev_img, step=16, scale=1.25)


# In[ ]:


if __name__ == '__main__':
    # Binarized (negative)
    prev_img = cv2.imread("/media/yahia/ballena/ABI/Atlantic/Averages_Binarized_Bilateral/Processed_algae_distribution_20220723.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/Atlantic/Averages_Binarized_Bilateral/Processed_algae_distribution_20220724.png")
    flow = deepflow(prev_img, next_img)
    plotly_flow_vectors(flow, prev_img, step=16, scale=1.25)
    # # Viridis
    # prev_img = cv2.imread("/media/yahia/ballena/ABI/Atlantic/Averages_Cropped/Processed_algae_distribution_20220723.png")
    # next_img = cv2.imread("/media/yahia/ballena/ABI/Atlantic/Averages_Cropped/Processed_algae_distribution_20220724.png")
    # flow = deepflow(prev_img, next_img)
    # plot_flow_vectors(flow, prev_img, step=16, scale=1.25)


# ### Partitioned

# In[ ]:


def process_region(lat_range, lon_range, lat_index, lon_index, n):
    prev_img_path = f'/media/yahia/ballena/ABI/Partition/n = {n}/Averages_Binarized_Bilateral/[{lat_range[0]},{lat_range[1]}],[{lon_range[0]},{lon_range[1]}]/Processed_algae_distribution_20220723.png'
    next_img_path = f'/media/yahia/ballena/ABI/Partition/n = {n}/Averages_Binarized_Bilateral/[{lat_range[0]},{lat_range[1]}],[{lon_range[0]},{lon_range[1]}]/Processed_algae_distribution_20220724.png'
    
    prev_img = cv2.imread(prev_img_path)
    next_img = cv2.imread(next_img_path)
    
    if prev_img is None or next_img is None:
        return lat_index, lon_index, None  # Skip this region if either image is not found
    
    flow = deepflow(prev_img, next_img)
    flow_image = plot_flow_vectors(flow, prev_img, step=16, scale=1.25, display=False)
    
    return lat_index, lon_index, flow_image


# In[ ]:


if __name__ == '__main__':
    n = 24
    lat_splits = np.linspace(12, 40, n + 1)
    lon_splits = np.linspace(-12, -100, n + 1)
    lat_splits = lat_splits.tolist()
    lon_splits = lon_splits.tolist()
    images_dict = {}
     
    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(len(lat_splits) - 1):
            for j in range(len(lon_splits) - 1):
                lat_range = (lat_splits[i], lat_splits[i + 1])
                lon_range = (lon_splits[j], lon_splits[j + 1])
                futures.append(executor.submit(process_region, lat_range, lon_range, i, j, n))
        
        for future in futures:
            i, j, flow_image = future.result()
            if flow_image is not None:
                images_dict[(i, j)] = flow_image
    
    # Initialize a list to store each row of stitched images
    stitched_rows = []
    
    for i in reversed(range(len(lat_splits) - 1)):  # Reverse the order of latitudes
        # Initialize a list to store images in the current row
        row_images = []
        
        for j in reversed(range(len(lon_splits) - 1)):  # Reverse the order of longitudes
            if (i, j) in images_dict:
                row_images.append(images_dict[(i, j)])
            else:
                # If any image is missing, create an empty placeholder of the same size
                # Assuming all images have the same dimensions
                row_images.append(np.zeros_like(images_dict[(0, 0)]))
        
        # Horizontally stack images in the current row
        stitched_row = np.hstack(row_images)
        stitched_rows.append(stitched_row)
    
    # Vertically stack all rows to create the final stitched image
    final_image = np.vstack(stitched_rows)
    
    # Save or display the final stitched image
    final_image_path = f'/media/yahia/ballena/TEST/test_{n}.png'
    cv2.imwrite(final_image_path, final_image)
    
    # Optionally display the final stitched image
    plt.imshow(final_image)
    plt.axis('off')
    plt.show()


# In[ ]:




