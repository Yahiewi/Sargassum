#!/usr/bin/env python
# coding: utf-8

# # Image Processing
# Using the results of the previous notebook, we're going to try to use the OpenCV library to process the images and save them so we can then apply our motion estimation algorithms to ABI-GOES aggregate images over a certain period of time (10 days perhaps).

# ## Importing necessary libraries and notebooks

# In[1]:


import xarray as xr
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from matplotlib import ticker
from IPython.display import Image, display, HTML
from multiprocessing import Pool
from scipy.ndimage import label

# Import the other notebooks without running their cells
from ii_Data_Manipulation import visualize_4
from iii_GOES_average import time_list, visualize_aggregate, calculate_median, split_and_aggregate_median, save_as_netcdf


# ## Preparing the Images
# We're going to work on 10 images obtained from averaging all ABI-GOES images for a given day. First, we need to average the images for each day and then save them to our hard drive.
# 
# One challenge is that acquisitions don't start and end at the same time for each day (acquisitions start at 12:00 for 2022/07/24 for example), so we need to be able to collect a list of the times at which we have data. 

# ### *collect_times*

# In[2]:


def collect_times(date, directory):
    """ Collect the earliest and latest acquisition times for a given date from file names. """
    prefix = f"cls-abi-goes-global-hr_1d_{date}"
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    times = [f.split('_')[-1].split('.')[0] for f in files]  # Assumes files are named '..._HH-MM.nc'
    if times:
        return min(times), max(times)
    return None


# ### *save_aggregate*
# Because we've encountered bugs and stack overflow when we tried to modify the function of notebook 3 by adding an optional parameter (**output_filepath**=None), which if specified saves the figure instead of showing it (and removes the legend), we've decided instead to write a new function here **save_aggregate** that can also display the image.

# In[3]:


def save_aggregate(aggregate_data, lat_range=None, lon_range=None, color="viridis", vmax=0.001, threshold=0.0001, output_filepath=None, netcdf_filepath=None, filter_clouds=True, display=False):
    # Select the desired subset
    if lat_range:
        aggregate_data = aggregate_data.sel(latitude=slice(*lat_range))
    if lon_range:
        aggregate_data = aggregate_data.sel(longitude=slice(*lon_range))

    # If filtering clouds, set NaN values to zero
    if filter_clouds:
        aggregate_data = xr.where(np.isnan(aggregate_data), 0, aggregate_data)

    # Set up a plot with geographic projections
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Customize the map with coastlines and features
    ax.coastlines(resolution='10m', color='black', visible=True)
    ax.add_feature(cfeature.BORDERS, linestyle=':', visible=True)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', visible=True)

    # Show gridlines only when visualizing interactively
    if display:
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        cbar_kwargs = {'shrink': 0.35}
    else:
        cbar_kwargs = None

    # Plot the aggregate data with the specified color, vmax, and threshold
    im = aggregate_data.plot(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree(),
                             cmap=color, add_colorbar=display,
                             vmin=threshold, vmax=vmax, cbar_kwargs=cbar_kwargs)

    # Set title and colorbar only when visualizing interactively
    if display:
        im.colorbar.set_label('Aggregate Floating Algae Index (FAI)')
        plot_date = aggregate_data.attrs.get('date', 'Unknown Date')
        plt.title(f"Aggregate Algae Distribution on {plot_date}")
        plt.show()

    if output_filepath:
        plt.savefig(output_filepath)  
        plt.close(fig)

    # Save the data as a NetCDF file if a filepath is provided
    if netcdf_filepath:
        # Create a new Dataset to include latitude and longitude
        dataset = xr.Dataset({
            'fai_anomaly': aggregate_data
        }, coords={
            'latitude': aggregate_data.latitude,
            'longitude': aggregate_data.longitude
        })
        dataset.to_netcdf(netcdf_filepath)
        plt.close(fig)


# We have the option to get the raw average (without the mask for the land) or to mask the land and we should test both images to see which is best for the OF algorithms.

# ### *process_dates*

# In[4]:


def process_dates(start_date, end_date, directory, output_dir, lat_range=None, lon_range=None, color="viridis", save_image=True, save_netcdf=False):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert the start and end dates from strings to datetime objects
    current_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    
    while current_date <= end_date:
        # Format the current date as a string in 'YYYYMMDD' format
        date_str = current_date.strftime('%Y%m%d')
        
        # Discover the start and end times of image acquisition on the current day by scanning the directory
        times = collect_times(date_str, directory)
        
        if times:
            # Create a list of timestamps for the day using the discovered start and end times
            times_for_day = time_list(
                datetime.strptime(date_str + '_' + times[0], '%Y%m%d_%H-%M'),
                datetime.strptime(date_str + '_' + times[1], '%Y%m%d_%H-%M'), 
                10  # Interval between images in minutes
            )
            
            # Calculate the median distribution of algae based on the list of timestamps
            median_distribution = calculate_median(times_for_day, lat_range, lon_range)
            
            # Prepare the output file paths for the current day's visualization and NetCDF file
            output_image_path = os.path.join(output_dir, f'algae_distribution_{date_str}.png') if save_image else None
            output_netcdf_path = os.path.join(output_dir, f'algae_distribution_{date_str}.nc') if save_netcdf else None
            
            # Visualize the median algae distribution and save it as both an image and NetCDF file (if the optional parameters are provided)
            save_aggregate(median_distribution, lat_range, lon_range, color=color, output_filepath=output_image_path, netcdf_filepath=output_netcdf_path, display=False)
        
        # Increment the current date by one day
        current_date += timedelta(days=1)


# In[5]:


# start_date = '20220723'
# end_date = '20220724'
# directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
# output_directory = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/ABI_Averages' 
# lat_range = (12, 17)  
# lon_range = (-67, -60) 

# # Call the function
# process_dates(start_date, end_date, directory, output_directory, lat_range, lon_range, save_image=True, save_netcdf=True)


# If we look at the images, we can see that some of them are covered by clouds which makes detecting the algae impossible. A solution we could implement is to use the OLCI images (if they're more clear) and add them to the ABI aggregates (using OpenCV's **OR** operator for example).

# ### *plot_xarray*
# Takes as input an xarray and outputs an interactive graph where you can hover over the pixels to get the corresponding value. Useful for debugging and testing.

# In[72]:


def plot_xarray(xarray_dataset, variable_name="fai_anomaly", title='Interactive Plot', xlabel='Longitude Index', ylabel='Latitude Index', cmap='viridis'):
    """
    Plot the specified variable from the given xarray dataset interactively.
    
    Parameters:
    xarray_dataset (xr.Dataset): The xarray dataset containing the data to plot.
    variable_name (str): The name of the variable to plot.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    cmap (str): The colormap to use for the plot.
    """
    get_ipython().run_line_magic('matplotlib', 'widget')
    
    xarray_data = xarray_dataset[variable_name]
    
    def plot_data(x=0, y=0):
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(xarray_data, origin='lower', cmap=cmap)
        
        # Title and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Text annotation for displaying value under cursor
        text = ax.text(0, 0, '', va='bottom', ha='left')
        
        def onclick(event):
            # Get the x and y pixel coords
            ix, iy = int(event.xdata + 0.5), int(event.ydata + 0.5)
            value = xarray_data.isel(longitude=ix, latitude=iy).values
            text.set_text(f'Value: {value}')
            text.set_position((ix, iy))
        
        fig.canvas.mpl_connect('button_press_event', onclick)
    
    interact(plot_data, x=(0, xarray_data.shape[1] - 1), y=(0, xarray_data.shape[0] - 1))


# ### Image Filters

# #### *display_image_mpl*

# In[1]:


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


# #### *crop_image*
# Let's write a function to crop images so as to remove the white space from the figure.

# In[6]:


def crop_image(image):
    # Convert to grayscale if it is a color image
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image

    # Threshold the image to isolate the content
    _, thresh = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours from the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour which will encompass the area of interest
        c = max(contours, key=cv2.contourArea)
        
        # Get the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(c)
        
        # Crop the original image using the dimensions of the bounding rectangle
        cropped_img = image[y:y+h, x:x+w]
        return cropped_img
    else:
        print("No significant contours found.")
        return image  # Return original image if no contours were found


# #### *binarize_image*
# Binarizing the images (indicating the presence of algae by absolute black and the rest by white) might be beneficial for our Optical Flow algorithms.

# In[2]:


def binarize_image(image, threshold):
    # Ensure the image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


# #### ~*binarize_otsu_image*~
# Automatic binarization using Otsu's method (which calculates an optimal threshold value).
# This returns a very noisy image in general and especially when applied to the atlantic NetCDF image.

# #### ~*binarize_adaptive_image*~
# Calculates thresholds for smaller regions of the image.
# Very aggressively filters the atlantic NetCDF image.

# #### *bilateral_image*

# In[22]:


def bilateral_image(image, diameter=9, sigmaColor=75, sigmaSpace=75):
    """
    Apply a bilateral filter to an image to reduce noise while keeping edges sharp.
    
    Parameters:
    - diameter (int): Diameter of each pixel neighborhood that is used during filtering.
                      If it is non-positive, it is computed from sigmaSpace.
    - sigmaColor (float): Filter sigma in the color space. A larger value of the parameter
                          means that farther colors within the pixel neighborhood (see sigmaSpace)
                          will be mixed together, resulting in larger areas of semi-equal color.
    - sigmaSpace (float): Filter sigma in the coordinate space. A larger value of the parameter
                          means that farther pixels will influence each other as long as their
                          colors are close enough (see sigmaColor). When d>0, it specifies the
                          neighborhood size regardless of sigmaSpace. Otherwise, d is proportional
                          to sigmaSpace.

    Returns:
    - filtered_image (ndarray): The image after applying the bilateral filter.
    """
    bilateral = cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
    return bilateral


# #### ~*edges*~

# #### ~*equalize_image*~
# This is an optional image processing step which should increase contrast in the image.

# #### *filter_by_size*
# This function filters out the small sargassum aggregates which might be noise.
# 
# **The smaller the threshold is, the slower it is. No longer.**

# In[3]:


def filter_by_size(image, size_threshold):
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    valid_labels = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= size_threshold]
    
    filtered_image = np.isin(labels, valid_labels).astype(np.uint8) * 255
    return filtered_image


# #### *adaptive_filter_by_size*
# Applies a different size threshold above a certain latitude (usually 30° N). This is done to filter more detections which are probably noise (in the north) and not filter out too much the the areas where there are genuine detections.

# In[40]:


def adaptive_filter_by_size(dataset, base_threshold, higher_threshold, latitude_limit=30):
    """
    Filters image components by size using xarray, applying a higher threshold for regions with latitude > 30.

    Args:
    dataset (xarray.Dataset): The input dataset containing the image data and latitude coordinate.
    base_threshold (int): The base threshold for the area of connected components.
    higher_threshold (int): The threshold for areas with latitude > latitude_limit.
    latitude_limit (float): The latitude above which the higher threshold is applied.

    Returns:
    xarray.Dataset: The dataset with the image data filtered by size.
    """
    # Ensure dataset has 'latitude' coordinate
    if 'latitude' not in dataset.coords:
        raise ValueError("Dataset must have 'latitude' as a coordinate")

    # Convert the data array to a numpy array
    image = dataset.to_array().values.squeeze()

    # Convert image to grayscale if it is not already
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Label the connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image.astype(np.uint8), connectivity=8)

    # Create an empty array to store the filtered image
    filtered_image = np.zeros_like(image, dtype=np.uint8)

    # Iterate over each component
    for i in range(1, num_labels):
        # Component stats
        _, y, _, h, area = stats[i]
        # Determine the latitude at the component's centroid
        centroid_latitude = dataset.latitude[y + h // 2].values

        # Apply thresholds based on latitude
        effective_threshold = higher_threshold if centroid_latitude > latitude_limit else base_threshold
        if area >= effective_threshold:
            filtered_image[labels == i] = 255

    # Convert the numpy array back to an xarray DataArray
    filtered_da = xr.DataArray(filtered_image, dims=dataset.dims, coords=dataset.coords)

    return filtered_da


# #### *adaptive_filter*
# Applies a filter based on local densities (the areas with more detections will be filtered less).

# In[44]:


def adaptive_filter(image, adaptive_base_threshold, window_size=10, density_scale_factor=1.5):
    """
    Adaptive filtering based on local densities.

    Args:
    image (numpy.ndarray): The input binary image.
    adaptive_base_threshold (int): The base threshold for the area of connected components.
    window_size (int): The size of the window to calculate local densities.
    density_scale_factor (float): Factor to scale the base threshold based on local density.

    Returns:
    numpy.ndarray: The filtered image.
    """
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate local density of detections
    kernel = np.ones((window_size, window_size), np.uint8)
    local_density = cv2.filter2D((image > 0).astype(np.uint8), -1, kernel) / (window_size**2)

    # Prepare output image
    output_image = np.zeros_like(image)

    # Label connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

    # Process each component
    for i in range(1, num_labels):
        # Get the component's stats
        x, y, w, h, area = stats[i]

        # Calculate adaptive threshold for the component
        local_density_mean = np.mean(local_density[y:y+h, x:x+w])
        adaptive_threshold = adaptive_base_threshold * (1 + local_density_mean * density_scale_factor)

        # Filter based on adaptive threshold
        if area >= adaptive_threshold:
            component_mask = (labels == i)
            output_image[component_mask] = 255

    return output_image


# #### *opening*
# An *erosion* followed by a *dilation*: first an erosion is applied to remove the small blobs, then a dilation is applied to regrow the size of the original object.
# 
# We can change the kernel_size as well as the kernel_shape. Increasing the kernel_size, increase the "aggression" of the filter while changing the kernel_shape didn't make much of a visible difference.

# In[8]:


def opening(image, kernel_size=3, kernel_shape=cv2.MORPH_RECT):
    # Convert to grayscale if the image is in color
    if len(image.shape) != 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create the structuring element (kernel)
    if kernel_shape == 'ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    elif kernel_shape == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    else:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply the morphological opening
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    return opened_image


# In[25]:


# # Opening Test
# if __name__ == "__main__":
#     # Paths
#     source_path = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages/algae_distribution_20220723.nc"
#     dest_path = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Opening_20220723.nc'
    
#     # Process the directory
#     process_netCDF(source_path, dest_path, threshold=1, bilateral=False, binarize=True, negative=False, median=False,
#                    filter_small=True, size_threshold=10, opened=True, kernel_size=2, land_mask=False, coast_mask=False, coast_threshold=50000)
#     # NOTE: if you get permission denied, don't forget to close ncviewer first


# #### ~*median_filter*~

# In[17]:


def median_filter(image, kernel_size=3):
    """
    Applies a median filter to an image to reduce noise.

    Parameters:
    - image (numpy.ndarray): The input image array, which should be a 2D grayscale image.
    - kernel_size (int): The size of the kernel used for the median filter. Must be an odd number.

    Returns:
    - numpy.ndarray: The image after applying the median filter.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to grayscale if the image is in color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply the median filter
    filtered_image = cv2.medianBlur(image, kernel_size)
    
    return filtered_image


# #### ~*anisotropic_diffusion*~
# Anisotropic diffusion works by encouraging diffusion in areas with small gradients and discouraging it across strong edges, thus preserving edges while reducing noise.

# The following function is used to create a NetCDF that will be used for mask_coast. (One-time use)

# In[110]:


# def create_reduced_netcdf(input_file, output_file):
#     # Load the original dataset
#     ds = xr.open_dataset(input_file)
    
#     # Extract the latitude and longitude values from gphit and glamt
#     latitude = ds['gphit'].values[:, 0]  # Assuming latitude varies along the first dimension
#     longitude = ds['glamt'].values[0, :]  # Assuming longitude varies along the second dimension
    
#     # Ensure unique and sorted values for coordinates
#     latitude = np.unique(latitude)
#     longitude = np.unique(longitude)
    
#     # Create a new DataArray for Distcoast with the new coordinates
#     distcoast = xr.DataArray(ds['Distcoast'].values, dims=('latitude', 'longitude'), coords={'latitude': latitude, 'longitude': longitude})
    
#     # Create a new dataset
#     new_ds = xr.Dataset({'Distcoast': distcoast})
    
#     # Save the new dataset to a NetCDF file
#     new_ds.to_netcdf(output_file)
    
#     return new_ds
    
# if __name__ == "__main__":
#     input_file = "/home/yahia/Documents/Jupyter/Sargassum/SARG12_distcoast.nc"
#     output_file = '/home/yahia/Documents/Jupyter/Sargassum/distcoast.nc'
#     new_dataset = create_reduced_netcdf(input_file, output_file)
#     print(new_dataset)


# #### *mask_coast*
# Function to filter detections that are close to the coast. Unlike the other functions, this one takes in a dataset as input not an image.

# In[7]:


def mask_coast(fai_dataset, distcoast_dataset_path='/home/yahia/Documents/Jupyter/Sargassum/Utilities/distcoast.nc', 
               threshold=5000, land_mask=True):
    # Load the distance from coast dataset
    distcoast_dataset = xr.open_dataset(distcoast_dataset_path)
    dist_from_coast = distcoast_dataset['Distcoast']
    
    # Ensure fai_dataset has 'latitude' and 'longitude' coordinates
    if 'latitude' not in fai_dataset.coords or 'longitude' not in fai_dataset.coords:
        fai_dataset = fai_dataset.rename({'y': 'latitude', 'x': 'longitude'})
    
    # Interpolate the distance from coast data to match fai_dataset's coordinate grid
    interpolated_dist_from_coast = dist_from_coast.interp(
        latitude=fai_dataset.latitude, 
        longitude=fai_dataset.longitude, 
        method='nearest'
    )
    
    # Define the boxes representing parts of North America
    box1 = (fai_dataset.longitude >= -100) & (fai_dataset.longitude <= -72) & (fai_dataset.latitude >= 28) & (fai_dataset.latitude <= 40)
    box2 = (fai_dataset.longitude >= -100) & (fai_dataset.longitude <= -79.68) & (fai_dataset.latitude >= 24) & (fai_dataset.latitude <= 28)
    box3 = (fai_dataset.longitude >= -100) & (fai_dataset.longitude <= -86) & (fai_dataset.latitude >= 17) & (fai_dataset.latitude <= 24)
    box4 = (fai_dataset.longitude >= -100) & (fai_dataset.longitude <= -79) & (fai_dataset.latitude >= 12) & (fai_dataset.latitude <= 17)
    
    # Combine the boxes into a single mask where True means it's within the NA boxes
    na_mask = box1 | box2 | box3 | box4
    
    # Mask where the distance is greater than or equal to the threshold OR outside NA boxes
    mask = (interpolated_dist_from_coast >= threshold) | ~na_mask
    
    # Optionally set land (distcoast == 0) to NaN
    if land_mask:
        land_mask = interpolated_dist_from_coast == 0
        interpolated_dist_from_coast = interpolated_dist_from_coast.where(~land_mask, other=np.nan)
        # Apply the mask to keep original values where the mask is True or set to 0 where False, and NaN for land
        masked_fai_dataset = fai_dataset.where(mask, other=0)
        masked_fai_dataset = masked_fai_dataset.where(~land_mask, other=np.nan)
    else:
        # Apply the mask without considering the land explicitly
        masked_fai_dataset = fai_dataset.where(mask, other=0)
    
    return masked_fai_dataset


# In[194]:


if __name__ == "__main__":
    fai_dataset = xr.open_dataset('/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages/algae_distribution_20220723.nc')
    threshold = 50000
    masked_fai_dataset = mask_coast(fai_dataset, threshold=threshold, land_mask=False)
    print(masked_fai_dataset)
    masked_fai_dataset.to_netcdf("/home/yahia/Documents/Jupyter/Sargassum/Images/Test/zboub.nc")
    # plot_xarray(masked_fai_dataset)


# ### *process_netCDF*
# Takes as input a netCDF file and applies the wanted filters to it and outputs another netCDF file.
# **Optimize the mask_coast part** (when both land_mask and coast_mask are false the function mask_coast will still be called but do nothing).

# In[48]:


def process_netCDF(
    source_path, dest_path=None, threshold=1, bilateral=False, binarize=False, 
    negative=False, filter_small=False, opened=False, kernel_size=3, size_threshold=50, 
    median=False, coast_mask=False, coast_threshold=5000, land_mask=False,
    adaptive=False, adaptive_base_threshold=20, window_size=10, density_scale_factor=1.5,
    adaptive_small=False, base_threshold=15, higher_threshold=50, latitude_limit=30):
    
    # Read the NetCDF file
    dataset = xr.open_dataset(source_path)
    
    # Extract the only variable in the dataset
    variable_name = list(dataset.data_vars)[0]
    data = dataset[variable_name].values

    # Ensure the data is 2D (grayscale image)
    if len(data.shape) == 3:
        data = data[0]  # Assuming the data is 3D and taking the first slice

    # Convert the data to an 8-bit image
    image = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Apply binary thresholding if required
    if binarize:
        image = binarize_image(image, threshold)

    # Make the image negative if required
    if negative:
        image = cv2.bitwise_not(image)

    # Filter small components if required
    if filter_small:
        image = filter_by_size(image, size_threshold)

    # Apply adaptive filter if required
    if adaptive:
        image = adaptive_filter(image, adaptive_base_threshold=adaptive_threshold, window_size=window_size, density_scale_factor=density_scale_factor)
    
    # Apply opening filter if required
    if opened:
        image = opening(image, kernel_size=kernel_size)

    # Apply bilateral filter if required
    if bilateral:
        image = bilateral_image(image)

    # Apply median filter if required
    if median:
        image = median_filter(image, kernel_size=3)

    # Convert image back to data array for further processing
    processed_data = xr.DataArray(image, dims=dataset[variable_name].dims, coords=dataset[variable_name].coords)

    # Reconstruct dataset with processed data for applying coastal mask
    processed_dataset = xr.Dataset({variable_name: processed_data})
    
    # Apply coastal mask if required (the else clause is to allow application of land_mask without coast_mask)
    if coast_mask:
        processed_dataset = mask_coast(processed_dataset, threshold=coast_threshold, land_mask=land_mask)

    else:
        processed_dataset = mask_coast(processed_dataset, threshold=0, land_mask=land_mask)

    if adaptive_small:
        processed_dataset = adaptive_filter_by_size(processed_dataset, base_threshold=base_threshold, higher_threshold=higher_threshold, latitude_limit=latitude_limit)

    # Save the processed data back to a new NetCDF file only if dest_path is specified
    if dest_path:
        processed_dataset.to_netcdf(dest_path)

    return processed_dataset


# In[51]:


# Atlantic Average
if __name__ == "__main__":
    # Paths
    source_path = "/media/yahia/ballena/ABI/NetCDF/Atlantic/Averages/algae_distribution_20220723.nc"
    dest_path = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Filtered_20220723.nc'
    
    # Process the directory
    process_netCDF(source_path, dest_path, threshold=1, bilateral=False, binarize=True, negative=False, 
                   filter_small=False, size_threshold=10, land_mask=False, coast_mask=False, coast_threshold=50000,
                   adaptive=False, adaptive_base_threshold=25, window_size=30, density_scale_factor=1.5,
                   adaptive_small=True, base_threshold=15, higher_threshold=10000, latitude_limit=30)
    # NOTE: if you get permission denied, don't forget to close ncviewer first


# Saving the result as a NetCDF with two variables (fai_anomaly and filtered).

# ### *process_directory_netCDF*
# Processes all the netCDF file in a given directory by calling the process_netCDF function on each one.

# In[42]:


def process_directory_netCDF(source_dir, dest_dir, threshold=9, bilateral=False, binarize=True, crop=False, negative=False):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.nc'):
            # Original NetCDF file path
            source_path = os.path.join(source_dir, filename)
            
            # New filename with 'Processed' prefix
            new_filename = 'Processed_' + filename
            
            # Define the output path for the processed NetCDF file
            dest_path = os.path.join(dest_dir, new_filename)
            
            # Process the NetCDF file
            process_netCDF(source_path, dest_path, threshold, bilateral, binarize, crop, negative)


# In[45]:


# if __name__ == '__main__':
#     source_dir = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/ABI_Averages/'
#     dest_dir = '/home/yahia/Documents/Jupyter/Sargassum/Images/Test/Processed_ABI_Averages'
#     process_directory_netCDF(source_dir, dest_dir, threshold=9, bilateral=False, binarize=True, crop=False, negative=False)


# ### *process_directory*
# This function takes as input images and applies the wanted functions to them and then saves them in the provided directory.

# In[ ]:


def process_directory(source_dir, dest_dir, threshold=180, bilateral=False, binarize=False, crop=True, negative=False):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.png'):
            # Original image path
            image_path = os.path.join(source_dir, filename)
            image = cv2.imread(image_path)
            
            # Filter the image
            if bilateral:
                image = bilateral_image(image)
            
            # Binarize the image
            if binarize:
                image = binarize_image(image, threshold)

            # Crop the image
            if crop:
                image = crop_image(image)

            # Make the image negative
            if negative:
                image = cv2.bitwise_not(image)
            
            # New filename with 'Processed' prefix
            new_filename = 'Processed_' + filename
            
            # Define the output path for the processed image
            output_path = os.path.join(dest_dir, new_filename)
            
            # Save the processed image
            cv2.imwrite(output_path, image)


# In[ ]:


# if __name__ == '__main__':
#     # Paths
#     source_directory = '/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages'
#     destination_directory = '/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral'
    
#     # Process the directory
#     process_directory(source_directory, destination_directory, threshold=100, bilateral=True, binarize=True)


# In[ ]:


# if __name__ == '__main__':
#     # Display the processed image
#     image_path = '/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Binarized_Bilateral/Binarized_Bilateral_algae_distribution_20220724.png'
#     display(Image(filename=image_path, width=700))  


# The **threshold** value must be chosen carefully so as to leave all the algae, but not leave the clouds, land or other undesirable features. 
# 
# This is what the Binarized version looks like (for **cmap="binary"** and **threshold=180**), this should make it easier for the OF algorithms to track the algae. If we increase the threshold (which leaves in more algae), we get a lot of discrete algae spots, which is probably not going to be good for our algorithms.

# In[ ]:


# if __name__ == '__main__':
#     image_path = '/home/yahia/Documents/Jupyter/Sargassum/Images/Binarized_algae_distribution_20220724_thresh_200.png'
#     display(Image(filename=image_path, width=700))  


# This is what the binarized version looks like for **threshold=200**.
# Maybe we could still use this, if we apply a filter to it (median filter for example).

# ## Finding a Good Example
# The 10-day period we have chosen (starting on 2022/07/24 and ending on 2022/08/02) may not be enough on its own to visualize motion vectors because a lot of the acquisitions are masked by clouds, so we're going to try to find 2 consecutive days in which the detections are clear after averaging the ABI-GOES images.
# 
# After a few tries, we've found the week of 2022/07/18 - 2022/07/24 to be good, with 22, 23 and 24 having particularly clear images.

# In[ ]:


if __name__ == '__main__':
    start_date = '20220718'
    end_date = '20220724'
    directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
    output_directory = '/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages' 
    latitude_range = (12, 17)  
    longitude_range = (-67, -60) 
    
    # Calculate the 1-day averages and save them
    process_dates(start_date, end_date, directory, output_directory, latitude_range, longitude_range, color="binary")
    
    # Paths
    source_directory = '/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages'
    destination_directory = '/home/yahia/Documents/Jupyter/Images/Sargassum/ABI_Averages_Binarized_Bilateral'
    
    # Process the directory (filter, binarize and crop the images)
    process_directory(source_directory, destination_directory, threshold=180, bilateral=True, binarize=True)


# ## Producing Viridis Images
# After all the image processing we did, the algorithm may not be able to track individual pixels any more, so raw viridis images may actually be better than the images we processed.

# In[ ]:


if __name__ == '__main__':
    start_date = '20220718'
    end_date = '20220724'
    directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
    output_directory = '/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Viridis' 
    latitude_range = (12, 17)  
    longitude_range = (-67, -60) 
    
    # Calculate the 1-day averages and save them
    process_dates(start_date, end_date, directory, output_directory, latitude_range, longitude_range, color="viridis")
    
    # Paths
    source_directory = '/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Viridis'
    destination_directory = '/home/yahia/Documents/Jupyter/Sargassum/Images/ABI_Averages_Processed_Viridis'
    
    # Process the directory (filter, binarize and crop the images)
    process_directory(source_directory, destination_directory, threshold=180, bilateral=False, binarize=False)


# # Producing the Databases

# ## ABI_Averages_Antilles
# We're going to average and process all the ABI-GOES images and save them to the directory ABI_Averages on the hard drive "ballena". Running this block might take a while. To optimize we could try and parallelize this process using the GPU.

# In[ ]:


if __name__ == '__main__':
    start_date = '20221121'
    end_date = '20221231'
    directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
    output_directory = '/media/yahia/ballena/ABI_Averages_Antilles' 
    latitude_range = (12, 17)  
    longitude_range = (-67, -60) 
    
    # Calculate the 1-day averages and save them
    process_dates(start_date, end_date, directory, output_directory, latitude_range, longitude_range, color="viridis")
    
    # Paths
    source_directory = '/media/yahia/ballena/ABI_Averages_Antilles' 
    destination_directory = '/media/yahia/ballena/ABI_Averages_Antilles_Processed' 
    
    # Process the directory (filter, binarize and crop the images)
    process_directory(source_directory, destination_directory, threshold=180, bilateral=False, binarize=False)


# In[ ]:


# Binarized and bilateral images
if __name__ == '__main__':
    # Paths
    source_directory = '/media/yahia/ballena/ABI/ABI_Averages_Antilles' 
    destination_directory = '/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral' 
    
    # Process the directory (filter, binarize and crop the images)
    process_directory(source_directory, destination_directory, threshold=100, bilateral=True, binarize=True)


# In[ ]:


# Binarized and bilateral images (negative)
if __name__ == '__main__':
    # Paths
    source_directory = '/media/yahia/ballena/ABI/ABI_Averages_Antilles' 
    destination_directory = '/media/yahia/ballena/ABI/ABI_Averages_Antilles_Binarized_Bilateral_Negative' 
    
    # Process the directory (filter, binarize and crop the images)
    process_directory(source_directory, destination_directory, threshold=100, bilateral=True, binarize=True, negative=True)


# ## MODIS_Images
# The function **process_dates** we previously defined is only adapted to ABI-GOES images, we will need to write a function that does the same for MODIS and OLCI images. We will also need to do the same for **save_aggregate**.

# In[ ]:


def save_image(file_path, lat_range=None, lon_range=None, color="viridis", vmax=0.1, output_filepath=None):
    # Load the netCDF data
    data = xr.open_dataset(file_path)
    
    # If ranges are specified, apply them to select the desired subset
    if lat_range:
        data = data.sel(latitude=slice(*lat_range))
    if lon_range:
        data = data.sel(longitude=slice(*lon_range))

    # Determine the index data and labels based on instrument used
    index_key = 'fai_anomaly' if "abi" in file_path else 'nfai_mean'
    colorbar_label = 'Floating Algae Index Anomaly (FAI)' if "abi" in file_path else 'Normalized Floating Algae Index (NFAI)'
    title = 'FAI anomaly across the selected region on ' if "abi" in file_path else 'NFAI across the selected region on '
    
    # Extract relevant data (NFAI or FAI anomaly)
    index_data = data[index_key]

    # Set non-positive values to a very small negative number, close to zero
    index_data = xr.where(index_data > 0, index_data, -0.1)
    
    # Set up a plot with geographic projections
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Customize the map with coastlines and features
    ax.coastlines(resolution='10m', color='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    # Show gridlines only when visualizing interactively, not when saving the output
    if output_filepath is None:
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        cbar_kwargs = {'shrink': 0.35}
    else:
        cbar_kwargs = None

    # Plot the data with the modified contrast
    im = index_data.plot(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree(),
                         cmap=color, add_colorbar=True, extend='both',
                         vmin=-0.01, vmax=vmax,  # Here we set the scale to max out at 0.5
                         cbar_kwargs={'shrink': 0.35})

    # Set title and colorbar only when visualizing interactively
    if output_filepath is None:
        im.colorbar.set_label('Normalized Floating Algae Index (NFAI)')
        plot_date = data.attrs.get('date', 'Unknown Date')
        plt.title(f"Algae Distribution on {plot_date}")

    if output_filepath:
        plt.savefig(output_filepath)  
        plt.close(fig)  
    else:
        plt.show()  


# In[ ]:


def process_dates_2(start_date, end_date, directory, output_dir, lat_range=None, lon_range=None, color="viridis"):
    # Convert the start and end dates from strings to datetime objects
    current_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    
    while current_date <= end_date:
        # Format the current date as a string in 'YYYYMMDD' format
        date_str = current_date.strftime('%Y%m%d')
        
        # Prepare the output file path for the current day's visualization
        # Visualize the median algae distribution and save it using the provided visualization function
        if "modis" in directory:
            output_file_path = os.path.join(output_dir, f'MODIS_{date_str}.png')
            file_path = directory + f"/cls-modis-aqua-global-lr_1d_{date_str}.nc"
        elif "olci" in directory:
            output_file_path = os.path.join(output_dir, f'OLCI_{date_str}.png')
            file_path = directory + f"/cls-olci-s3-global-lr_1d_{date_str}.nc"

        # Check if the file exists before proceeding
        if not os.path.exists(file_path):
            print(f"File not found for date: {date_str}, skipping...")
        else:
            try:
                save_image(file_path, lat_range, lon_range, color=color, output_filepath=output_file_path)
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
    
        # Increment the current date by one day
        current_date += timedelta(days=1)


# Generating the MODIS images:

# In[ ]:


if __name__ == '__main__':
    start_date = '20201207'
    end_date = '20221231'
    directory = '/media/yahia/ballena/CLS/modis-aqua-global-lr' 
    output_directory = '/media/yahia/ballena/MODIS_Antilles' 
    latitude_range = (12, 17)  
    longitude_range = (-67, -60) 
    
    # Calculate the 1-day averages and save them
    process_dates_2(start_date, end_date, directory, output_directory, latitude_range, longitude_range, color="viridis")
    
    # Paths
    source_directory = '/media/yahia/ballena/MODIS_Antilles' 
    destination_directory = '/media/yahia/ballena/MODIS_Antilles_Processed' 
    
    # Process the directory (filter, binarize and crop the images)
    process_directory(source_directory, destination_directory, threshold=180, bilateral=False, binarize=False)


# ## OLCI_Images

# Generating the OLCI images:

# In[ ]:


if __name__ == '__main__':
    start_date = '20201207'
    end_date = '20240122'
    directory = '/media/yahia/ballena/CLS/olci-s3-global-lr' 
    output_directory = '/media/yahia/ballena/OLCI_Antilles' 
    latitude_range = (12, 17)  
    longitude_range = (-67, -60) 
    
    # Calculate the 1-day averages and save them
    process_dates_2(start_date, end_date, directory, output_directory, latitude_range, longitude_range, color="viridis")
    
    # Paths
    source_directory = '/media/yahia/ballena/OLCI_Antilles' 
    destination_directory = '/media/yahia/ballena/OLCI_Antilles_Processed' 
    
    # Process the directory (filter, binarize and crop the images)
    process_directory(source_directory, destination_directory, threshold=180, bilateral=False, binarize=False)


# In[ ]:


### ATLANTIC TEST


# In[ ]:


# if __name__ == '__main__':
#     start_date = '20220723'
#     end_date = '20220724'
#     directory = '/media/yahia/ballena/CLS/abi-goes-global-hr' 
#     output_directory = '/media/yahia/ballena/TEST/Atlantic' 
#     lat_splits = [12, 16, 20, 24, 28, 32, 36, 40]  # Define latitude splits
#     lon_splits = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -12]  # Define longitude splits

#     def process_dates_3(start_date, end_date, directory, output_dir, lat_splits, lon_splits, color="viridis"):
#         current_date = datetime.strptime(start_date, '%Y%m%d')
#         end_date = datetime.strptime(end_date, '%Y%m%d')
        
#         while current_date <= end_date:
#             date_str = current_date.strftime('%Y%m%d')
#             times = collect_times(date_str, directory)
            
#             if times:
#                 times_for_day = time_list(
#                     datetime.strptime(f'{date_str}_{times[0]}', '%Y%m%d_%H-%M'),
#                     datetime.strptime(f'{date_str}_{times[1]}', '%Y%m%d_%H-%M'),
#                     interval=10
#                 )
#                 median_distribution = split_and_aggregate_median(lat_splits, lon_splits, times_for_day)
#                 output_file_path = os.path.join(output_dir, f'algae_distribution_{date_str}.png')
#                 save_aggregate(median_distribution, color=color, output_filepath=output_file_path)
            
#             current_date += timedelta(days=1)

#     # Calculate the 1-day averages and save them
#     process_dates_3(start_date, end_date, directory, output_directory, lat_splits, lon_splits, color="viridis")
    
#     # Paths
#     source_directory = '/media/yahia/ballena/TEST/Atlantic' 
#     destination_directory = '/media/yahia/ballena/Test/Atlantic_Cropped' 
    
#     # Process the directory (filter, binarize and crop the images)
#     process_directory(source_directory, destination_directory, threshold=180, bilateral=False, binarize=False)


# Saving the image after calculating the median on each region then combining gives the same exact result as calculating the median over the whole region.

# In[ ]:




