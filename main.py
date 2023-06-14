from __future__ import absolute_import, division, print_function
from mayavi import mlab
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import normalize
import pyproj
import pptk
from ripser import ripser
from ripser import Rips
from datetime import datetime
from scipy.spatial import distance_matrix

# Converts longitude or latitude values to meters
def to_meters(longs_or_lats):
    return longs_or_lats * 111139.0

# Define the projection map
proj = pyproj.Proj(proj='utm', zone=50, ellps='WGS84')

# Define the DataFrame
df = pd.read_csv("dataset_raw_full.csv")

# Create new column with encoding for each label in the dataframe
labels_cat = pd.Categorical(df.Label)
df['Encoding'] = labels_cat.codes

# Create dictionary with (key=label, value=encoding)
# {'taxi': 7, 'walk': 9, 'bus': 3, 'train': 8, 'car': 4, 'airplane': 0,
# 'subway': 6, 'bike': 1, 'run': 5, 'boat': 2}
labels = df['Label'].drop_duplicates()
encodings = df['Encoding'].drop_duplicates()
label_map = dict(zip(labels, encodings))

# Filter the data on latitude and longitude to obtain only the points in centre/suburbs
lat_long_mask = (df['Longitude'] > 115.5) & (df['Longitude'] < 116.7) & (df['Latitude'] > 39.5) & (df['Latitude'] < 40.25)
df = df[lat_long_mask]

# Filter the data on daytime traffic
df['Date_Time'] = pd.to_datetime(df['Date_Time'])
daytime_mask = (df['Date_Time'].dt.hour >= 7) & (df['Date_Time'].dt.hour < 8) # from 07.00 to 08.00
df = df[daytime_mask]

# Filter the data on car traffic
car_encoding = label_map['car']
car_mask = (df['Encoding'] == car_encoding)
df = df[car_mask]

df['Long_meters'] = to_meters(df['Longitude'].values)
df['Lat_meters'] = to_meters(df['Latitude'].values)

# Visualize the filtered points
x, y = proj(df['Longitude'].tolist(), df['Latitude'].tolist())
p = np.c_[x, y, 0.3048 * df['Altitude']] # convert alt to meters
v = pptk.viewer(p)













# # Construct the point cloud
# point_cloud = np.column_stack((x_coords, y_coords))
#
# # Compute the persistence diagram
# diagrams = ripser(point_cloud)['dgms']
#
# # Print the persistence diagrams
# for diagram in diagrams:
#     print(diagram)
