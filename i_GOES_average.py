#!/usr/bin/env python
# coding: utf-8

# # Averaging ABI-GOES images for a given day:
# We could try to average the images for a given day to try and reproduce the images we see on the CLS datastore.
# GOES doesn't cover the same region from image to image, so we can't directly calculate the average, we'll have to calculate an average only when there is data (non-nan values). We're going to work on the same day of 24/07/2022.

# ## Importing necessary libraries and notebooks

# In[1]:


#%matplotlib widget
import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from matplotlib import ticker
from IPython.display import Image
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


# ## time_list
# First, we should write a function that generates a lost of the times in the time intervals we need to make the importation of data easier. 

# In[2]:


def time_list(start_time, end_time, interval):
    """
    Generate a list of datetime strings in the format 'YYYYMMDD_HH-MM' between start_time and end_time at intervals of 'interval' minutes.
    
    Parameters:
    - start_time (datetime): The start time.
    - end_time (datetime): The end time.
    - interval (int): The interval in minutes between each time point.

    Returns:
    - times (list of str): List of formatted datetime strings.
    """
    
    # Generate a list of times at the specified interval
    times = []
    current_time = start_time
    while current_time <= end_time:
        times.append(current_time.strftime('%Y%m%d_%H-%M'))
        current_time += timedelta(minutes=interval)

    return times


# ## visualize_4
# Note: vmax doesn't set a threshold for the image, it's just that the colors are saturated at vmax (For example if vmax is 0.01, values greater than 0.01 will still be shown but will have the same saturated color).
# 
# This function is taken from the now deleted notebook ii_Data_Manipulation.ipynb

# In[ ]:


def visualize_4(file_path, lat_range=None, lon_range=None, color="viridis", vmax=0.1):
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

    # Adding grid lines and disabling labels on the top and right
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Plot the data with the modified contrast
    im = index_data.plot(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree(),
                         cmap=color, add_colorbar=True, extend='both',
                         vmin=-0.01, vmax=vmax,  # Here we set the scale to max out at 0.5
                         cbar_kwargs={'shrink': 0.35})

    # Add color bar details
    im.colorbar.set_label(colorbar_label)
    
    # Show the plot with title
    plt.title(title + str(data.time.values[0]))
    plt.show()


# ## visualize_aggregate
# We should first write a function **(very similar to visualize_5, maybe we should make it use visualize_5)** to visualize the aggregate motion of the algae, this function would take the aggregate_data we're going to calculate as argument instead of the path to the file.

# In[3]:


def visualize_aggregate(aggregate_data, lat_range=None, lon_range=None, color="viridis", vmax=0.001, threshold=0, output_filepath=None, filter_clouds=True):
    # Select the desired subset
    if lat_range:
        aggregate_data = aggregate_data.sel(latitude=slice(*lat_range))
    if lon_range:
        aggregate_data = aggregate_data.sel(longitude=slice(*lon_range))
    
    # If filtering clouds, set NaN values to -0.1
    if filter_clouds:
        aggregate_data = xr.where(np.isnan(aggregate_data), -0.1, aggregate_data)
    
    # Set up a plot with geographic projections
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Customize the map with coastlines and features
    ax.coastlines(resolution='10m', color='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    # Adding grid lines and disabling labels on the top and right
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Plot the aggregate data with the specified color, vmax, and threshold
    im = aggregate_data.plot(ax=ax, x='longitude', y='latitude', transform=ccrs.PlateCarree(),
                             cmap=color, add_colorbar=True, extend='both',
                             vmin=threshold, vmax=vmax, cbar_kwargs={'shrink': 0.35})

    # Add color bar details
    colorbar_label = 'Aggregate Floating Algae Index (FAI)' 
    im.colorbar.set_label(colorbar_label)
    
    # Show the plot with title
    plt.title("Aggregate Algae Distribution on 2022-07-24")
    plt.show()


# ## save_as_netcdf

# In[11]:


def save_as_netcdf(dataset, output_filepath):
    """
    Save the given Dataset to a NetCDF file.

    Parameters:
    - dataset (Dataset): The xarray Dataset to save.
    - output_filepath (str): The path to the output NetCDF file.
    """
    dataset.to_netcdf(output_filepath)


# We obtain what appear to be the same results whether we calculate the median on the whole image, then zoom in, or zoom in then calculate it.

# ## Calculations

# ### calculate_median

# In[12]:


def calculate_median(time_list, lat_range=None, lon_range=None, threshold=0):
    """
    Calculate the median of algae presence over a given time range based on a list of times,
    within specified latitude and longitude ranges.

    Parameters:
    - time_list (list of str): List of formatted datetime strings in the format 'YYYYMMDD_HH-MM'.
    - lat_range (tuple): Tuple of (min_latitude, max_latitude).
    - lon_range (tuple): Tuple of (min_longitude, max_longitude).
    - threshold (float): The threshold above which data is considered.

    Returns:
    - median_algae_distribution (DataArray): The median algae distribution within the specified region.
    """
    aggregate_data_list = []

    # Loop over each time in the time list, loading the data and adding it to the list
    for time_str in time_list:
        file_path = f"/media/yahia/ballena/CLS/abi-goes-global-hr/cls-abi-goes-global-hr_1d_{time_str}.nc"
        # Skip if the file does not exist
        if not os.path.exists(file_path):
            print(f"Skipping: {file_path} does not exist.")
            continue
        
        data = xr.open_dataset(file_path)

        # Apply latitude and longitude ranges if specified
        if lat_range:
            data = data.sel(latitude=slice(*lat_range))
        if lon_range:
            data = data.sel(longitude=slice(*lon_range))

        # Extract the index of interest and drop the 'time' coordinate
        algae_data = data['fai_anomaly'].squeeze(drop=True)

        # Mask the data to include only algae (values greater than the threshold)
        algae_masked = algae_data.where(algae_data > threshold)

        # Add the masked data to our list (each element in this list is the data array, after processing, for the give time)
        aggregate_data_list.append(algae_masked)

    # Combine the data along a new dimension, then calculate the mean along that dimension
    # Note: Xarray's mean function by default ignores nan values
    aggregate_data = xr.concat(aggregate_data_list, dim='new_dim')
    median_algae_distribution = aggregate_data.median(dim='new_dim')

    # Extract the date from the first time string and set it as an attribute (Used for the figure title)
    date_from_time = time_list[0].split('_')[0]  # Assuming time_list items are 'YYYYMMDD_HH-MM'
    median_algae_distribution.attrs['date'] = date_from_time

    return median_algae_distribution


# In[ ]:


if __name__ == '__main__':
    # Generating the time list
    times = time_list(start_time=datetime(2022, 7, 24, 12, 0), end_time=datetime(2022, 7, 24, 18, 50), interval=10)
    
    # Calculating the median data for this time period
    median_algae_distribution = calculate_median(times,lat_range=(14, 15), lon_range= (-66, -65))
    
    # Calculating the aggregate data for this time period
    average_algae_distribution = calculate_mean(times,lat_range=(12, 17), lon_range=(-67, -60))
    
    #Visualizing the result and comparing it to the mean 
    visualize_aggregate(median_algae_distribution, (14, 15), (-66, -65), color="viridis", vmax=0.001, threshold=0)
    visualize_aggregate(average_algae_distribution, (12, 17), (-67, -60), color="viridis", vmax=0.001, threshold=0)


# Although the difference is not very big, it is non negligible and we can see that median function produces rafts that are a bit thinner, which is preferable.

# In[ ]:


if __name__ == '__main__':#
    # Generating the time list
    times = time_list(start_time=datetime(2022, 7, 24, 12, 0), end_time=datetime(2022, 7, 24, 18, 50), interval=10)
    
    # Calculating the min data for this time period
    min_algae_distribution = calculate_min(times,lat_range=(12, 17), lon_range=(-67, -60))
    
    # Calculating the mean data for this time period
    average_algae_distribution = calculate_mean(times,lat_range=(12, 17), lon_range=(-67, -60))
    
    #Visualizing the result and comparing it to the mean
    visualize_aggregate(min_algae_distribution, (12, 17), (-67, -60), color="viridis", vmax=0.001, threshold=0)
    visualize_aggregate(average_algae_distribution, (12, 17), (-67, -60), color="viridis", vmax=0.001, threshold=0)


# In[ ]:


# TEST


# In[13]:


def calculate_median_n(times, lat_range=None, lon_range=None, threshold=0):
    """
    Calculate the median of algae presence over a given time range based on a list of times,
    within specified latitude and longitude ranges.

    Parameters:
    - time_list (list of str): List of formatted datetime strings in the format 'YYYYMMDD_HH-MM'.
    - lat_range (tuple): Tuple of (min_latitude, max_latitude).
    - lon_range (tuple): Tuple of (min_longitude, max_longitude).
    - threshold (float): The threshold above which data is considered.

    Returns:
    - median_dataset (Dataset): The median algae distribution within the specified region.
    """
    aggregate_data_list = []

    # Loop over each time in the time list, loading the data and adding it to the list
    for time_str in times:
        file_path = f"/media/yahia/ballena/CLS/abi-goes-global-hr/cls-abi-goes-global-hr_1d_{time_str}.nc"
        # Skip if the file does not exist
        if not os.path.exists(file_path):
            print(f"Skipping: {file_path} does not exist.")
            continue
        
        data = xr.open_dataset(file_path)

        # Apply latitude and longitude ranges if specified
        if lat_range:
            data = data.sel(latitude=slice(*lat_range))
        if lon_range:
            data = data.sel(longitude=slice(*lon_range))

        # Extract the index of interest and drop the 'time' coordinate
        algae_data = data['fai_anomaly'].squeeze(drop=True)

        # Mask the data to include only algae (values greater than the threshold)
        algae_masked = algae_data.where(algae_data > threshold)

        # Add the masked data to our list (each element in this list is the data array, after processing, for the give time)
        aggregate_data_list.append(algae_masked)

    # Combine the data along a new dimension, then calculate the median along that dimension
    # Note: Xarray's median function by default ignores nan values
    aggregate_data = xr.concat(aggregate_data_list, dim='new_dim')
    median_algae_distribution = aggregate_data.median(dim='new_dim')

    # Create a new Dataset to include latitude and longitude
    median_dataset = xr.Dataset({
        'median_fai_anomaly': median_algae_distribution
    }, coords={
        'latitude': median_algae_distribution.latitude,
        'longitude': median_algae_distribution.longitude
    })

    # Extract the date from the first time string and set it as an attribute (Used for the figure title)
    date_from_time = times[0].split('_')[0]  # Assuming time_list items are 'YYYYMMDD_HH-MM'
    median_dataset.attrs['date'] = date_from_time

    return median_dataset


# In[ ]:


if __name__ == "__main__" :
    # Generating the time list
    times = time_list(start_time=datetime(2022, 7, 24, 12, 0), end_time=datetime(2022, 7, 24, 18, 50), interval=10)
    
    # Calculate median
    median_dataset = calculate_median(times, lat_range=(12, 17), lon_range=(-67, -60), threshold=0)
    
    # Save to NetCDF
    output_filepath = '/home/yahia/Documents/Jupyter/Sargassum/median_algae_distribution.nc'
    save_as_netcdf(median_dataset, output_filepath)
    
    print(f"Median algae distribution saved to {output_filepath}")

