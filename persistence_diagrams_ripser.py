from __future__ import absolute_import, division, print_function
from mayavi import mlab
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import normalize
from ripser import ripser
from ripser import Rips
import pyproj
import pptk
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry import Point


# Define the projection map
proj = pyproj.Proj(proj='utm', zone=50, ellps='WGS84')
# Define the DataFrame
df = pd.read_csv("filtered_car_trajectories.csv")

# Create coordinate projection
x, y = proj(df['Longitude'].tolist(), df['Latitude'].tolist())
p = np.c_[x, y, 0.3048 * df['Altitude']] # convert alt to meters
v = pptk.viewer(p)


# Create persistence diagrams
x_coords = df['Longitude'].tolist()
y_coords = df['Latitude'].tolist()

# Construct the point cloud
print("Constructing point cloud")
point_cloud = np.column_stack((x_coords, y_coords))
print(f"Amount of points: {len(point_cloud)}")

north = 40.25
east = 116.7
south = 39.5
west = 115.5

outer_districts = ['Changping District', 'Shunyi District', 'Tongzhou District', 'Daxing District', 'Fangshan District', 'Mentougou District']
inner_districts = ['Haidian District', 'Chaoyang District', 'Fengtai District', 'Shijingshan District']
city_centre = ['Dongcheng District', 'Xicheng District']


def plot_map(points):
    x, y = proj([point[0] for point in points], [point[1] for point in points])
    p = np.c_[x, y, np.zeros_like(x)]
    return pptk.viewer(p)


# Returns the points within a given district area (outer, inner, city centre)
def points_within_district(districts):
    district_polygons = []
    for district in districts:
        query = district + ', Beijing, China'
        polygon = ox.geocode_to_gdf(query)
        district_polygons.append(polygon)

    district_points = []

    # Iterate over the points and check if they belong to one of the districts
    for point in point_cloud:
        point_obj = Point(point[0], point[1])

        for i, polygon in enumerate(district_polygons):
            if polygon['geometry'].iloc[0].contains(point_obj):
                district_points.append(point)
                break

    return district_points


outer_district_points = points_within_district(outer_districts)
inner_district_points = points_within_district(inner_districts)
city_centre_points = points_within_district(city_centre)

plot_map(outer_district_points)
plot_map(inner_district_points)
plot_map(city_centre_points)

