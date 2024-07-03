#!/usr/bin/env python
# coding: utf-8

# # Flow_Analysis
# This notebook contains the functions to analyze the results of the OF algorithms (DeepFlow for the moment). This includes calculating the velocity of the algae, image segmentation (averaging the vectors to calculate velocity of each raft), surface area calculation, interactive plots, etc.

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

# Import the other notebooks without running their cells
from ii_Data_Manipulation import visualize_4
from iii_GOES_average import time_list, visualize_aggregate, calculate_median
from iv_Image_Processing import collect_times, crop_image, save_aggregate, binarize_image, bilateral_image, process_dates, process_directory
from v_i_OF_Functions import *


# ## Quantitative Analysis
# Here we're going to try to visualize the magnitude of the displacement vectors in meters to be able to judge whether the flow our algorithm is physically consistent or not.

# ### ~*calculate_velocity*~

# In[ ]:


def calculate_velocity(flow, resolution_km=1, time_seconds=24*3600):
    # Calculate the magnitude of the flow vectors
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    # Convert from pixels to meters (1 km = 1000 meters)
    magnitude_meters = magnitude * resolution_km * 1000
    
    # Calculate velocity in meters per second
    velocity_m_per_s = magnitude_meters / time_seconds
    return velocity_m_per_s


# ### *visualize_velocity*

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

# ### *segment_aggregations*

# In[ ]:


def segment_aggregations(mask):
    # Ensure mask is binary and of type uint8
    mask = (mask * 255).astype(np.uint8)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


# ### *calculate_average_vectors*

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


# ### *haversine*
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


# ### *calculate_area_haversine*
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


# ### *generate_lat_lon_arrays*

# In[ ]:


def generate_lat_lon_arrays(lat_range, lon_range, image_shape):
    """
    Generate latitude and longitude arrays for the given image shape and coordinate ranges.
    """
    latitudes = np.linspace(lat_range[0], lat_range[1], image_shape[0])
    longitudes = np.linspace(lon_range[0], lon_range[1], image_shape[1])
    lats, lons = np.meshgrid(latitudes, longitudes, indexing='ij')
    return lats, lons


# ### *calculate_velocity_and_angles*
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


# ### *calculate_angular_velocity*
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


# ### *plot_aggregations_with_vectors*

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

# ### *plot_interactive_contours_with_vectors*

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

