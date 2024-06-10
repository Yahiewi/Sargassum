#!/usr/bin/env python
# coding: utf-8

# # Optical Flow Functions
# In this notebook, we're going to define the functions we will need for the display, calculations, GIF creation and other utilities that are useful (these functions were previously defined in v_Motion_Estimation but we moved them here to avoid circular dependencies).

# ## Importing necessary libraries and notebooks

# In[ ]:


import xarray as xr
import io
import os
import cv2
import imageio
import plotly.graph_objects as go
import mpld3
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from IPython.display import Image, display, HTML
from PIL import Image as PILImage
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from plotly.subplots import make_subplots
from geopy.distance import geodesic
from math import radians, sin, cos, sqrt, atan2
import matplotlib as mpl
# Increase the embed limit for animations
mpl.rcParams['animation.embed_limit'] = 50  # Increase the limit to 50 MB

# Import the other notebooks without running their cells
from ii_Data_Manipulation import visualize_4
from iii_GOES_average import time_list, visualize_aggregate, calculate_median
from iv_Image_Processing import collect_times, crop_image, save_aggregate, binarize_image, bilateral_image, process_dates, process_directory


# ## Optical Flow Algorithms

# ### Farneback_flow

# In[ ]:


def farneback_flow(prev_img, next_img):
    """
    Returns:
    - flow : np.ndarray
        The computed flow image that will have the same size as `prev_img` and
        type CV_32FC2. Each element of the flow matrix will be a vector that
        indicates the displacement (in pixels) of the corresponding pixel from
        the first image to the second image.

    Method Parameters:
    - flow : np.ndarray
        Optional input flow estimate. It must be a single precision floating point
        image with the same size as `prev_img`. If provided, the function uses it as
        an initial approximation of the flow. If None, the function estimates the flow
        from scratch.
    - pyr_scale : float
        The image scale (<1) to build pyramids for each image; pyr_scale=0.5
        means a classical pyramid, where each next layer is twice smaller than
        the previous one.
    - levels : int
        The number of pyramid layers including the initial image. Levels=1
        means that no extra layers are created and only the original images are used.
    - winsize : int
        The size of the window used to smooth derivatives used as a basis
        for the polynomial expansion. The larger the size, the smoother the
        input image and the more robust the algorithm is to noise, but the more
        blurred motion details become.
    - iterations : int
        The number of iterations the algorithm will perform at each pyramid level.
        More iterations can improve the accuracy of the flow estimation.
    - poly_n : int
        The size of the pixel neighborhood used to find polynomial expansion
        in each pixel. Typical values are 5 or 7.
    - poly_sigma : float
        The standard deviation of the Gaussian that is used to smooth derivatives
        used as a basis for the polynomial expansion. This parameter can
        typically be ~1.1 for poly_n=5 and ~1.5 for poly_n=7.
    - flags : int
        Operation flags that can specify extra options such as using the initial
        flow estimates or applying a more sophisticated form of smoothing:
        - cv2.OPTFLOW_USE_INITIAL_FLOW: Uses the input flow as an initial flow estimate.
        - cv2.OPTFLOW_FARNEBACK_GAUSSIAN: Uses a Gaussian window for smoothing
          derivatives instead of a box filter.
    """
    # Make images grayscale
    prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    next_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_img, next_img, flow = None, pyr_scale=0.5, levels=3, winsize=15, 
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow


# ### Lucas-Kanade

# In[ ]:


def LK_flow(prev_img, next_img, max_corners=100, quality_level=0.3, min_distance=7, block_size=7, win_size=(15, 15), max_level=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)):
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


# ### ~Lucas-Kanade Flow~
# This is a version that returns the flow like the Farneback method.

# In[ ]:


def LK_flow_2(prev_img, next_img):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Detecting good features to track
    prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, qualityLevel=0.01, minDistance=7, blockSize=7)
    
    # Calculate optical flow using Lucas-Kanade method
    next_points, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_points, None, **lk_params)
    
    # Select good points
    good_prev = prev_points[st == 1]
    good_next = next_points[st == 1]

    # Create flow array
    flow = np.zeros((prev_img.shape[0], prev_img.shape[1], 2), dtype=np.float32)
    for pt1, pt2 in zip(good_prev, good_next):
        x1, y1 = pt1.ravel()
        x2, y2 = pt2.ravel()
        flow[int(y1), int(x1)] = (x2 - x1, y2 - y1)

    return flow


# ### DeepFlow

# In[ ]:


def deepflow(prev_img, next_img):
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    
    # Initialize DeepFlow
    deep_flow = cv2.optflow.createOptFlow_DeepFlow()
    
    # Compute flow
    flow = deep_flow.calc(prev_gray, next_gray, None)
    
    return flow


# ## Useful Functions

# ### compute_flow_components
# Computes the magnitude and angle of the optical flow from the given flow vector components and then visualizes them.

# In[ ]:


def compute_flow_components(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    return mag, ang


# ### visualize_flow_components
# Visualizes the magnitude and angle of optical flow using matplotlib.

# In[ ]:


def visualize_flow_components(mag, ang):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Magnitude plot
    ax[0].imshow(mag, cmap='hot')
    ax[0].set_title('Optical Flow Magnitude')
    ax[0].axis('off')

    # Angle plot
    # Normalize the angles between 0 and 1 for visualization
    ang_normalized = ang / (2 * np.pi)
    ax[1].imshow(ang_normalized, cmap='hsv')  # HSV colormap to represent angle
    ax[1].set_title('Optical Flow Angle')
    ax[1].axis('off')

    plt.show()


# ### plot_flow_vectors
# We can also visualize the motion field through vectors. This uses quiver from matplotlib.
# 
# Quiver produces nice looking arrows, but for our purposes, overlay_flow_vectors is probably better.

# In[ ]:


def plot_flow_vectors(flow, base_img, step=16, scale=1, display=True, color='r'):
    """
    Creates a plot of optical flow vectors over the base image and optionally displays it.

    :param flow: Computed flow vectors with shape (H, W, 2).
    :param base_img: Base image on which to plot the vectors.
    :param step: Grid step size for sampling vectors. Smaller values increase density.
    :param scale: Scaling factor for the magnitude of vectors to enhance visibility.
    :param display: Boolean indicating whether to display the plot.
    :return: An image array of the plot.
    """
    H, W = flow.shape[:2]
    y, x = np.mgrid[0:H:step, 0:W:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(base_img, cmap='gray')  # Ensure the image is displayed correctly
    ax.quiver(x, y, fx, fy, color=color, angles='xy', scale_units='xy', scale=1/scale, width=0.0025)
    ax.set_xlim([0, W])
    ax.set_ylim([H, 0])
    ax.axis('off')  # Turn off the axis

    # Optionally display the plot
    if display:
        plt.show()

    # Convert the Matplotlib figure to a PIL Image and then to a NumPy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)
    img = PILImage.open(buf)
    img_arr = np.array(img)

    return img_arr


# In[ ]:


# if __name__ == '__main__':
#     prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220723.png")
#     next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png")
#     flow = farneback_flow(prev_img, next_img)
#     plot_flow_vectors(flow, prev_img, step=16, scale=1.25)
#     # display(Image(filename="/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png", width =750))


# ### plotly_flow_vectors

# In[ ]:


def plotly_flow_vectors(flow, base_img, step=16, scale=1, arrow_sampling=5):
    """
    Enhanced to show arrowheads selectively for performance.

    :param arrow_sampling: Only plot an arrowhead for every nth vector.
    """
    H, W = flow.shape[:2]
    y, x = np.mgrid[0:H:step, 0:W:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T * scale

    fig = go.Figure()
    fig.add_trace(go.Image(z=base_img))

    # Add fewer arrowheads based on sampling rate
    for i, (xi, yi, fxi, fyi) in enumerate(zip(x, y, fx, fy)):
        fig.add_trace(
            go.Scatter(
                x=[xi, xi + fxi],
                y=[yi, yi + fyi],
                mode='lines',
                line=dict(color='red', width=2),
                showlegend=False
            )
        )
        if i % arrow_sampling == 0:  # Only add arrowheads for every nth vector
            fig.add_annotation(
                x=xi + fxi,
                y=yi + fyi,
                ax=xi,
                ay=yi,
                xref='x',
                yref='y',
                axref='x',
                ayref='y',
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red'
            )

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, W]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[H, 0], scaleanchor="x", scaleratio=1),
        width=800,
        height=800,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    fig.show()


# In[ ]:


# if __name__ == '__main__':
#     # default
#     prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220723.png")
#     next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png")
#     flow = farneback_flow(prev_img, next_img)
#     plot_flow_vectors(flow, prev_img, step=16, scale=1.25)
#     # plotly
#     prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220723.png")
#     next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png")
#     flow = farneback_flow(prev_img, next_img)
#     plotly_flow_vectors(flow, prev_img, step=16, scale=1.25)


# ### plot_flow_vectors_opencv

# In[ ]:


def plot_flow_vectors_opencv(flow, base_img, step=8, scale=1, color=(0, 0, 255), max_height=800):
    """
    Draws optical flow vectors over the base image using OpenCV and displays the result.
    Closes the display window when 'q' is pressed.

    :param flow: Computed flow vectors with shape (H, W, 2).
    :param base_img: Base image on which to plot the vectors.
    :param step: Grid step size for sampling vectors. Smaller values increase density.
    :param scale: Scaling factor for the magnitude of vectors to enhance visibility.
    :param color: Color of the vectors in BGR format (default is red).
    """
    # Check if image is grayscale and convert to color
    if len(base_img.shape) == 2 or base_img.shape[2] == 1:
        base_img_color = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        base_img_color = base_img.copy()

    H, W = flow.shape[:2]
    y, x = np.mgrid[0:H:step, 0:W:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T * scale

    # Draw arrows or lines representing the flow vectors
    for xi, yi, fxi, fyi in zip(x, y, fx, fy):
        end_point = (int(xi + fxi), int(yi + fyi))
        cv2.arrowedLine(base_img_color, (xi, yi), end_point, color, thickness=1, tipLength=0.2)  
    
    # # Resize the image for display if it's too large
    # if H > max_height:
    #     scale_factor = max_height / H
    #     new_width = int(W * scale_factor)
    #     resized_img = cv2.resize(base_img_color, (new_width, max_height))
    # else:
    #     resized_img = base_img_color
    
    # Display the image with flow vectors
    cv2.imshow('Optical Flow Vectors', base_img_color)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


# In[ ]:


if __name__ == '__main__':
    # default
    prev_img = cv2.imread("/media/yahia/ballena/ABI/Atlantic/Averages_Binarized_Bilateral/Processed_algae_distribution_20220723.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/Atlantic/Averages_Binarized_Bilateral/Processed_algae_distribution_20220724.png")
    flow = deepflow(prev_img, next_img)
    plot_flow_vectors_opencv(flow, prev_img, step=16, scale=1.25)


# In[ ]:


if __name__ == '__main__':
    # default
    prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220723.png")
    next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png")
    flow = deepflow(prev_img, next_img)
    plot_flow_vectors_opencv(flow, prev_img, step=16, scale=1.25)


# ### overlay_flow_vectors
# Overlays optical flow vectors on an image and returns the resulting image with vectors. Uses arrowedLine from OpenCV.

# In[ ]:


def overlay_flow_vectors(flow, base_img, step=16, scale=1, color=(255, 0, 0)):
    # Ensure base_img is in RGB to display colored vectors
    if len(base_img.shape) == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
    
    H, W = flow.shape[:2]
    y, x = np.mgrid[0:H:step, 0:W:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    # Create a figure for drawing
    result_img = np.copy(base_img)
    for i in range(len(x)):
        start_point = (x[i], y[i])
        end_point = (int(x[i] + fx[i] * scale), int(y[i] + fy[i] * scale))
        cv2.arrowedLine(result_img, start_point, end_point, color, 1, tipLength=0.3)

    return result_img


# ### overlay_flow_vectors_with_quiver

# In[ ]:


def overlay_flow_vectors_with_quiver(flow, base_img, step=16, scale=1, color='r'):
    """
    Overlays flow vectors using matplotlib's quiver directly on the image,
    ensuring that the quiver plot matches the image size and aspect ratio.
    """
    if len(base_img.shape) == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)

    H, W = base_img.shape[:2]  # Use image dimensions for scaling and grid generation
    y, x = np.mgrid[0:H:step, 0:W:step].reshape(2, -1)
    fx, fy = flow[y, x].T

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)  # Set figure size based on image dimensions
    ax.imshow(base_img, extent=[0, W, H, 0], aspect='auto')  # Force aspect ratio to match the image
    ax.quiver(x, y, fx, fy, color=color, angles='xy', scale_units='xy', scale=1/scale, width=0.002)
    ax.set_xlim([0, W])
    ax.set_ylim([H, 0])
    ax.axis('off')

    # Convert figure to an image
    canvas = FigureCanvas(fig)
    canvas.draw()
    img_rgba = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    img = img_rgba.reshape(canvas.get_width_height()[::-1] + (4,))[..., :3]  # Convert ARGB to RGB

    plt.close(fig)
    return img


# ### create_flow_gif
# We can try to visualize the result using a GIF.

# In[ ]:


def create_flow_gif(images, gif_path, fps=1, loop=10, quiver=False):
    """
    Creates a GIF from a sequence of images, calculating optical flow and overlaying vectors.
    """
    images_for_gif = []
    
    for i in range(len(images) - 1):
        prev_img = images[i]
        next_img = images[i+1]
        flow = farneback_flow(prev_img, next_img)  # Assumes existence of this function
        
        if quiver:
            overlay_img = overlay_flow_vectors_with_quiver(flow, prev_img)
        else:
            overlay_img = overlay_flow_vectors(flow, prev_img)  # Assumes existence of this function
        
        images_for_gif.append(prev_img)  # Add original image
        images_for_gif.append(overlay_img)  # Add image with vectors

    # Add the last image to the gif
    images_for_gif.append(images[-1])

    # Write GIF
    imageio.mimsave(gif_path, images_for_gif, fps=fps, loop=loop)


# In[ ]:


# if __name__ == '__main__':
#     # Saving the GIF
#     prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220723.png")
#     next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png")
#     images = [prev_img, next_img]
#     create_flow_gif(images, '/home/yahia/Documents/Jupyter/Sargassum/Images/GIFs/optical_flow.gif', fps=0.2, loop=10, quiver=True)
    
#     # Displaying the GIF
#     gif_path = '/home/yahia/Documents/Jupyter/Sargassum/Images/GIFs/optical_flow.gif' 
#     display(Image(filename=gif_path))


# ### plot_LK_vectors
# This function plots the flow vectors using the results of the LK algorithm.

# In[ ]:


def plot_LK_vectors(p0, p1, st, base_img, display=True, color='r'):
    """
    Creates a plot of optical flow vectors over the base image and optionally displays it.

    :param p0: Initial points in the previous image.
    :param p1: Corresponding points in the next image.
    :param st: Status array indicating whether the flow for the corresponding feature has been found.
    :param base_img: Base image on which to plot the vectors.
    :param display: Boolean indicating whether to display the plot.
    :param color: Color of the flow vectors.
    :return: An image array of the plot.
    """
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(base_img, cmap='gray')  # Ensure the image is displayed correctly

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        ax.plot([c, a], [d, b], color=color, linewidth=1.5)
        ax.scatter(a, b, color=color, s=5)

    ax.axis('off')

    # Optionally display the plot
    if display:
        plt.show()

    # Convert the Matplotlib figure to a PIL Image and then to a NumPy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)
    img = PILImage.open(buf)
    img_arr = np.array(img)

    return img_arr


# ### display_image_cv

# In[ ]:


def display_image_cv(image_array):
    # OpenCV might load images in BGR format, ensure to convert to RGB if necessary
    if image_array.shape[2] == 3:  # Color image
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    cv2.imshow('Optical Flow Vectors', image_array)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


# In[ ]:


# if __name__ == '__main__':
#     # Binary image
#     prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220723.png")
#     next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png")
#     p0, p1, st, err = LK_flow(prev_img, next_img)
#     img_with_vectors = LK_vector_field(p0, p1, st, prev_img)
#     display_image_cv(img_with_vectors)


# ### display_image_mpl

# In[ ]:


def display_image_mpl(image_array, scale=1):
    """
    Displays an image using matplotlib. Converts from BGR to RGB if needed and handles both grayscale and color images.
    Allows specification of the display size.

    Parameters:
    - image_array (numpy array): The image data array. It can be in grayscale or BGR color format.
    - width (float): Width of the figure in inches.
    - height (float): Height of the figure in inches.
    """
    # Check if image is in color (BGR format), and convert to RGB for display
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # Create a figure with specified size
    plt.figure(figsize=(8*scale, 6*scale))
    
    # Determine if the image is grayscale and display it
    if len(image_array.shape) == 2 or image_array.shape[2] == 1:
        plt.imshow(image_array, cmap='gray')  # Display grayscale image
    else:
        plt.imshow(image_array)  # Display color image
    
    # Hide axes and show the figure
    plt.axis('off')
    plt.show()


# ### superpose_images
# This is a function that takes two images (preferably binarized for clarity) and superposes them on top of each other with different colors.

# In[ ]:


def superpose_images(image1, image2, color1=(255, 0, 0), color2=(0, 255, 0)):
    """
    Superposes the black regions of two binarized images onto a white background with different colors.
    Black areas from image1 and image2 are shown in distinct colors, and the background remains white.

    Parameters:
    - image1 (numpy.ndarray): The first binarized image, white background with black algae.
    - image2 (numpy.ndarray): The second binarized image, white background with black algae.
    - color1 (tuple): RGB color for the algae in the first image.
    - color2 (tuple): RGB color for the algae in the second image.

    Returns:
    - numpy.ndarray: An image with the black regions of the two input images superposed in the specified colors.
    """
    # Ensure images are grayscale
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Threshold images to ensure they are binary
    _, image1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
    _, image2 = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)

    # Create a white RGB image with the same dimensions as the input images
    height, width = image1.shape
    colored_image = np.full((height, width, 3), fill_value=(255, 255, 255), dtype=np.uint8)  # White background

    # Apply the specified colors to the black regions of each binary image
    colored_image[(image1 == 0)] = color1  # Apply color1 where image1 is black
    colored_image[(image2 == 0)] = color2  # Apply color2 where image2 is black

    return colored_image


# In[ ]:


if __name__ == '__main__':
    prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220723.png")
    next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png")
    superposed = superpose_images(prev_img, next_img)
    display_image_cv(superposed)


# In[ ]:


if __name__ == '__main__':
    motion_field = overlay_flow_vectors(flow, superposed, step=16, scale=1, color=(0,0,255))
    display_image_cv(motion_field)


# ### warp_image

# In[ ]:


def warp_image(img, flow):
    """
    Warps an image using the given optical flow map.

    Parameters:
    - img (numpy.ndarray): The original image to be warped.
    - flow (numpy.ndarray): The optical flow vectors that indicate pixel displacements.

    Returns:
    - warped_img (numpy.ndarray): The resulting image after applying the flow warp.
    """
    h, w = img.shape[:2]
    # Create grid of coordinates
    flow_map = np.column_stack((np.indices((h, w))[1].ravel() + flow[..., 0].ravel(),  # x coordinates
                                np.indices((h, w))[0].ravel() + flow[..., 1].ravel())) # y coordinates
    # Map coordinates from flow
    flow_map = flow_map.reshape(h, w, 2).astype(np.float32)
    # Apply remapping
    warped_img = cv2.remap(img, flow_map, None, cv2.INTER_LANCZOS4)

    return warped_img


# In[ ]:


if __name__ == '__main__':
    prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220723.png")
    next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220724.png")
    flow = farneback_flow(prev_img, next_img)
    #plot_of_vectors(flow, prev_img, step=16, scale=1.25)
    warped = warp_image(prev_img, flow)
    display_image_mpl(warped)
    # display(Image(filename="/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220723.png", width =750))
    # display(Image(filename="/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220724.png", width =750))


# ### warp_image_2

# In[ ]:


def warp_image_2(img, flow, alpha):
    h, w = flow.shape[:2]
    flow = flow * alpha
    
    # Create a grid of coordinates and apply the flow
    coords = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.array(coords).astype(np.float32)
    coords = np.stack(coords, axis=-1)
    coords += flow
    
    # Warp the image using the flow
    map_x = coords[..., 0].astype(np.float32)
    map_y = coords[..., 1].astype(np.float32)
    warped_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped_img


# ### interpolate_images
# We're now going to try and visualize the movement of the algae from one frame to the next by interpolating between the frames. We're first going to try a linear interpolation method that simply divides the flow into **num_interpolations** fields. This function then applies a fraction of the flow to the first image and (the opposite of that flow) to the second image then blends them together.

# In[ ]:


def interpolate_images(prev_img, next_img, flow, num_interpolations=30):
    interpolated_images = []
    for i in range(num_interpolations + 1):
        alpha = i / num_interpolations
        warped_prev = warp_image_2(prev_img, flow, alpha)
        warped_next = warp_image_2(next_img, -flow, 1 - alpha)
        blended_img = cv2.addWeighted(warped_prev, 1 - alpha, warped_next, alpha, 0)
        interpolated_images.append(blended_img)
    return interpolated_images


# ### visualize_movement

# In[ ]:


def visualize_movement(interpolated_images, fps=15):
    interval = 1000 / fps  # Interval in milliseconds

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(interpolated_images[0])
    
    def update_frame(num):
        im.set_array(interpolated_images[num])
        return im,
    
    ani = animation.FuncAnimation(fig, update_frame, frames=len(interpolated_images), blit=True, interval=interval)
    
    # Display the animation in the notebook
    display(HTML(ani.to_jshtml()))
    plt.close(fig)


# In[ ]:


# if __name__ == '__main__':
#     # Binarized
#     prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220723.png")
#     next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png")
#     flow = farneback_flow(prev_img, next_img)
#     interpolated_images = interpolate_images(prev_img, next_img, flow, num_interpolations=60)
#     visualize_movement(interpolated_images, fps=15)
#     #display(Image(filename='interpolated_images.gif'))


# In[ ]:


if __name__ == '__main__':
    # Viridis
    prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220723.png")
    next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220724.png")
    flow = farneback_flow(prev_img, next_img)
    interpolated_images = interpolate_images(prev_img, next_img, flow, num_interpolations=60)
    visualize_movement(interpolated_images, fps=10)


# In[ ]:


# Lucas-Kanade
if __name__ == '__main__':
    # Viridis
    prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220723.png")
    next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220724.png")
    flow = LK_flow_2(prev_img, next_img)
    interpolated_images = interpolate_images(prev_img, next_img, flow, num_interpolations=60)
    visualize_movement(interpolated_images, fps=10)


# ## Error Quantification

# ### calculate_mse

# In[ ]:


def calculate_mse(image1, image2):
    """
    Calculates the Mean Squared Error between two images, which measures the average of the squares of the errors.
    
    Parameters:
    - image1 (numpy.ndarray): The first image (e.g., warped image).
    - image2 (numpy.ndarray): The second image (e.g., actual next frame).

    Returns:
    - float: The mean squared error between the two images.
    """
    # Ensure the images are the same shape
    assert image1.shape == image2.shape, "Images must have the same dimensions."

    # Compute the squared differences
    diff = np.square(image1.astype("float") - image2.astype("float"))
    
    # Return the mean of the differences
    mse = np.mean(diff)
    return mse


# In[ ]:


if __name__ == '__main__':
    prev_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220723.png")
    next_img = cv2.imread("/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis/Processed_algae_distribution_20220724.png")
    flow = farneback_flow(prev_img, next_img)
    #plot_of_vectors(flow, prev_img, step=16, scale=1.25)
    warped = warp_image(prev_img, flow)
    display_image_mpl(warped)
    print("MSE = " + str(calculate_mse(warped, next_img)))


# In[ ]:





# ## Masking
# In this part, we're going to try and mask the parts which don't interest us (where there is no movement of algae). We'll start with the binarized image. The idea is to combine the two masks for the white pixels in the first and second image using the **bitwise OR operator**.
# 
# We thought of combining both the previous and next image in the mask because it seems like the flow vectors can trace out the shape of the algae in the next image, but when we tried, we found that masking using only the first image makes more sense.

# ### mask_flow_vectors
# Similar to plot_flow_vectors with the addition of the Mask optional parameter.

# In[ ]:


def mask_flow_vectors(flow, prev_img, combined_img, step=16, scale=1.25):
    h, w = prev_img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y, x].T
    
    mask = combined_img[y, x] > 0  # Only consider white areas
    x = x[mask]
    y = y[mask]
    fx = fx[mask]
    fy = fy[mask]

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(prev_img, cmap='gray')  # Ensure the image is displayed correctly
    ax.quiver(x, y, fx, fy, color='r', angles='xy', scale_units='xy', scale=1/scale, width=0.0025)
    ax.set_xlim([0, w])
    ax.set_ylim([h, 0])
    ax.axis('off')  # Turn off the axis
    plt.show()


# In[ ]:


# Zoom
if __name__ == '__main__':
    # Binarized
    prev_img = cv2.imread("/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral_Binarized_Bilateral/Processed_algae_distribution_20220723.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral_Binarized_Bilateral/Processed_algae_distribution_20220724.png")
    combined_img = cv2.bitwise_or(prev_img, prev_img)  # Combine masks to include both positions, use prev_img, next_img to combine
    combined_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2GRAY)
    flow = deepflow(prev_img, next_img)
    mask_flow_vectors(flow, prev_img, combined_img, step=16, scale=1.25)


# In[ ]:


# 23/07 - 24/07
if __name__ == '__main__':
    # Binarized
    prev_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral/Processed_algae_distribution_20220723.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral/Processed_algae_distribution_20220724.png")
    combined_img = cv2.bitwise_or(prev_img, prev_img)  # Combine masks to include both positions, use prev_img, next_img to combine
    combined_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2GRAY)
    flow = deepflow(prev_img, next_img)
    plot_flow_vectors(flow, prev_img, step=16, scale=1.25)
    mask_flow_vectors(flow, prev_img, combined_img, step=16, scale=1.25)


# We can see that the results aren't great when we try to apply this to an image with a bigger range (zoomed out).
# 
# We could try to change the scale of the arrows, or write another algorithm with a less aggressive mask that doesn't mask vectors in the close vicinity of the white pixels.

# ## Quantitative Analysis
# Here we're going to try to visualize the magnitude of the displacement vectors in meters to be able to judge whether the flow our algorithm is physically consistent or not.

# ### calculate_velocity

# In[ ]:


def calculate_velocity(flow, resolution_km=1, time_seconds=24*3600):
    # Calculate the magnitude of the flow vectors
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    # Convert from pixels to meters (1 km = 1000 meters)
    magnitude_meters = magnitude * resolution_km * 1000
    
    # Calculate velocity in meters per second
    velocity_m_per_s = magnitude_meters / time_seconds
    return velocity_m_per_s


# ### visualize_velocity

# In[ ]:


def visualize_velocity(velocity, prev_img):
    plt.figure(figsize=(10, 10))
    plt.imshow(prev_img, cmap='gray', alpha=0.5)
    plt.imshow(velocity, cmap='jet', alpha=0.5)
    plt.colorbar(label='Velocity (m/s)')
    plt.title('Flow Velocity Heatmap')
    plt.show()


# In[ ]:


# Zoom
if __name__ == '__main__':
    prev_img = cv2.imread("/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral_Binarized_Bilateral/Processed_algae_distribution_20220723.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral_Binarized_Bilateral/Processed_algae_distribution_20220724.png") 
    flow = deepflow(prev_img, next_img)
    plot_flow_vectors(flow, prev_img, step=16, scale=1.25)
    
    # Calculate the velocity of the flow vectors
    velocity_m_per_s = calculate_velocity(flow, resolution_km=1)
    print("Flow Velocity (m/s):", velocity_m_per_s)
    visualize_velocity(velocity_m_per_s, prev_img)

    # Masked Version
    prev_img = cv2.imread("/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral_Binarized_Bilateral/Processed_algae_distribution_20220723.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral_Binarized_Bilateral/Processed_algae_distribution_20220724.png")
    prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    flow = deepflow(prev_img, next_img)
    mask_flow_vectors(flow, prev_img, prev_img, step=16, scale=1.25)


# ## Image Segmentation
# Here we'll try to calculate the average of the vectors on each algae aggregate so as to be able to visualize and quantify the movement of the whole aggregate instead of individual pixels.

# ### segment_aggregations

# In[ ]:


def segment_aggregations(mask):
    # Ensure mask is binary and of type uint8
    mask = (mask * 255).astype(np.uint8)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


# ### calculate_average_vectors

# In[ ]:


def calculate_average_vectors(flow, contours):
    avg_vectors = []
    for contour in contours:
        mask = np.zeros(flow.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        masked_flow = flow[mask == 1]
        if masked_flow.size != 0:  # Ensure there are vectors to average
            avg_vector = masked_flow.mean(axis=0)
            avg_vectors.append(avg_vector)
        else:
            avg_vectors.append(None)  # No valid vectors in this contour
    return avg_vectors


# ### haversine
# This function was tested and returns correct results (distance in km)

# In[ ]:


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between two points 
    on the Earth specified by their longitude and latitude.
    """
    # Convert latitude and longitude from degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r


# ### calculate_area_haversine
# Since the distances are not uniform in the image we can't directly use **cv2.contourArea** to calculate the area as this function calculates it in terms of pixels.

# In[ ]:


def calculate_area_haversine(contour, lons, lats):
    """
    Calculate the area of the contour using the Haversine formula.
    """
    contour = contour.reshape(-1, 2)
    points = [(lats[y, x], lons[y, x]) for x, y in contour]
    centroid = np.mean(points, axis=0)

    area_km2 = 0.0
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        a = haversine(p1[0], p1[1], centroid[0], centroid[1])
        b = haversine(p2[0], p2[1], centroid[0], centroid[1])
        c = haversine(p1[0], p1[1], p2[0], p2[1])
        s = (a + b + c) / 2
        area_km2 += sqrt(s * (s - a) * (s - b) * (s - c))

    return area_km2


# ### generate_lat_lon_arrays

# In[ ]:


def generate_lat_lon_arrays(lat_range, lon_range, image_shape):
    """
    Generate latitude and longitude arrays for the given image shape and coordinate ranges.
    """
    latitudes = np.linspace(lat_range[0], lat_range[1], image_shape[0])
    longitudes = np.linspace(lon_range[0], lon_range[1], image_shape[1])
    lats, lons = np.meshgrid(latitudes, longitudes, indexing='ij')
    return lats, lons


# ### calculate_velocity_and_angles
# A generalization of the calculate_velocity function defined above.

# In[ ]:


def calculate_velocity_and_angle(vector, lon1, lat1, lon2, lat2, time_seconds=24*3600):
    """
    Calculate the velocity magnitude and angle between two points defined by longitude and latitude.
    """
    distance_km = haversine(lon1, lat1, lon2, lat2)
    magnitude_meters = distance_km * 1000
    velocity_m_per_s = magnitude_meters / time_seconds
    
    # Calculate the angle of the vector (in degrees)
    angle_degrees = np.degrees(np.arctan2(vector[1], vector[0]))
    
    return velocity_m_per_s, angle_degrees


# ### calculate_angular_velocity
# Since we're not considering individual pixels in this section, we need the angular velocity along with the regular velocity to be able to describe the movement of the contour.
# 
# **Right now, this function returns absurd results**.

# In[ ]:


def calculate_angular_velocity(flow, contour, centroid, resolution_km=1, time_seconds=24*3600, units='degrees'):
    mask = np.zeros(flow.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
    y, x = np.where(mask == 1)
    
    angular_velocity = 0
    for (i, j) in zip(x, y):
        vector = flow[j, i]
        r_vector = np.array([i - centroid[0], j - centroid[1]])
        cross_product = np.cross(r_vector, vector)
        angular_velocity += cross_product

    # Normalize by the number of points and convert to radians per second
    if len(x) > 0:
        angular_velocity /= len(x)
        angular_velocity_meters = angular_velocity * resolution_km * 1000
        angular_velocity_radians_per_s = angular_velocity_meters / time_seconds
    else:
        angular_velocity_radians_per_s = 0
    
    if units == 'degrees':
        angular_velocity_units_per_s = np.degrees(angular_velocity_radians_per_s)
    else:
        angular_velocity_units_per_s = angular_velocity_radians_per_s
    
    return angular_velocity_units_per_s


# ### plot_aggregations_with_vectors

# In[ ]:


def plot_aggregations_with_vectors(img, contours, avg_vectors):
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for proper color display
    
    for idx, (contour, vector) in enumerate(zip(contours, avg_vectors)):
        M = cv2.moments(contour)
        color = colors[idx % len(colors)]
        if M['m00'] != 0 and vector is not None:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            ax.quiver(cx, cy, vector[0], vector[1], color=color, scale=1.5, angles='xy', scale_units='xy')
            ax.text(cx, cy, str(idx), color=color, fontsize=12, verticalalignment='bottom')  # Label the vector
            # Draw the contour
            contour = contour.reshape(-1, 2)
            ax.plot(contour[:, 0], contour[:, 1], color=color, linewidth=2)
            # Label the contour
            centroid_x, centroid_y = np.mean(contour, axis=0).astype(int)
            ax.text(centroid_x, centroid_y, str(idx), color=color, fontsize=12, verticalalignment='top')  # Label the contour
        else:
            print(f"Contour {idx} with no area or no valid vectors found.")
    
    plt.show()


# In[ ]:


if __name__ == "__main__":
    # Load images
    prev_img = cv2.imread("/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral_Binarized_Bilateral/Processed_algae_distribution_20220723.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral_Binarized_Bilateral/Processed_algae_distribution_20220724.png")
    
    # Calculate flow
    flow = deepflow(prev_img, next_img)
    
    # Create a mask for the algae aggregation (example mask creation)
    mask = (prev_img[:,:,0] > 200) & (prev_img[:,:,1] > 200) & (prev_img[:,:,2] > 200)
    
    # Segment the algae aggregations
    contours = segment_aggregations(mask)

    # Calculate average vectors for each aggregation
    avg_vectors = calculate_average_vectors(flow, contours)
    
    # Plot the aggregations with their average vectors
    plot_aggregations_with_vectors(prev_img, contours, avg_vectors)


# ## Interactive Plot
# The idea here is to be able to hover with the mouse over a contour and see the corresponding velocity.

# ### plot_interactive_contours_with_vectors

# In[ ]:


def plot_interactive_contours_with_vectors(img, contours, avg_vectors, flow, lons, lats, grid_step=10):
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow']
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Image(z=rgb_img), row=1, col=1)

    height, width = img.shape[:2]

    # Reverse the latitude array to ensure correct orientation
    lats = np.flipud(lats)

    # Create a scatter plot for the image with hover information at a lower resolution
    hover_data = []
    for y in range(0, height, grid_step):
        for x in range(0, width, grid_step):
            hover_data.append((x, y, lats[y, x], lons[y, x]))

    hover_x, hover_y, hover_lat, hover_lon = zip(*hover_data)

    fig.add_trace(go.Scatter(
        x=hover_x,
        y=hover_y,
        mode='markers',
        marker=dict(size=1, opacity=0),
        hoverinfo='text',
        text=[f"Lat: {lat:.6f}, Lon: {lon:.6f}" for lat, lon in zip(hover_lat, hover_lon)],
        showlegend=False
    ))

    for idx, (contour, vector) in enumerate(zip(contours, avg_vectors)):
        if vector is not None:
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            lon1, lat1 = lons[cy, cx], lats[cy, cx]
            lon2, lat2 = lons[cy + int(vector[1]), cx + int(vector[0])], lats[cy + int(vector[1]), cx + int(vector[0])]
            velocity_m_per_s, angle_degrees = calculate_velocity_and_angle(vector, lon1, lat1, lon2, lat2)
            contour = contour.reshape(-1, 2)
            x_coords = contour[:, 0]
            y_coords = contour[:, 1]
            color = colors[idx % len(colors)]
            
            # Calculate the area using Haversine formula
            area_km2 = calculate_area_haversine(contour, lons, lats)
            
            hovertext = (f"Contour {idx}<br>"
                         f"Velocity: {velocity_m_per_s:.2f} m/s<br>"
                         f"Angle: {angle_degrees:.2f} degrees<br>"
                         f"Lat: {lat1:.6f}, Lon: {lon1:.6f}<br>"
                         f"Surface area: {area_km2:.6f} km²")
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords, mode='lines+markers',
                line=dict(color=color), marker=dict(size=2), name=f'Contour {idx}',
                hovertext=hovertext, showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[cx, cx + vector[0]], y=[cy, cy + vector[1]], 
                mode='lines+markers', line=dict(color=color), 
                marker=dict(size=2), name=f'Vector {idx}', 
                hovertext=hovertext, showlegend=False
            ))
            fig.add_annotation(
                x=cx + vector[0], y=cy + vector[1], ax=cx, ay=cy,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=2, arrowwidth=1, arrowcolor=color
            )

    fig.update_layout(title='Algae Aggregations with Average Vectors', showlegend=False, 
                      xaxis=dict(visible=False), yaxis=dict(visible=False), 
                      width=1000, height=800)
    fig.show()


# In[ ]:


if __name__ == "__main__":
    # Load images
    prev_img = cv2.imread("/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral_Binarized_Bilateral/Processed_algae_distribution_20220723.png")
    next_img = cv2.imread("/media/yahia/ballena/ABI/Spiral/ABI_Averages_Spiral_Binarized_Bilateral/Processed_algae_distribution_20220724.png")
    
    # Calculate flow
    flow = deepflow(prev_img, next_img)
    
    # Create a mask for the algae aggregation (example mask creation)
    mask = (prev_img[:,:,0] > 200) & (prev_img[:,:,1] > 200) & (prev_img[:,:,2] > 200)
    
    # Segment the algae aggregations
    contours = segment_aggregations(mask)
    
    # Calculate average vectors for each aggregation
    avg_vectors = calculate_average_vectors(flow, contours)
    
    # Define latitude and longitude ranges
    lat_range = (14, 15)
    lon_range = (-66, -65)
    
    # Get the shape of the image
    image_shape = prev_img.shape[:2]
    
    # Generate latitude and longitude arrays
    lats, lons = generate_lat_lon_arrays(lat_range, lon_range, image_shape)
    
    # Plot the aggregations with their average vectors
    plot_interactive_contours_with_vectors(prev_img, contours, avg_vectors, flow, lons, lats)


# In[ ]:




