from __future__ import absolute_import, division, print_function
from mayavi import mlab
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import normalize
from ripser import ripser
import pyproj
import pptk
import ripserplusplus as rpp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def ripscomplex(X, hdim=1):
    pairs= rpp.run("--format point-cloud --dim "+str(hdim), X)[hdim]
    return np.sqrt(np.array((pairs.tolist())).reshape((len(pairs),2)))


def plot_dgm(dgm_true, dgm_tot, m, M, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    x, y = dgm_true[:,0], dgm_true[:,1]
    ax.scatter(x, y, marker='o', color='blue', label='true dgm')
    ax.scatter(dgm_tot[:,0], dgm_tot[:,1], marker='x', color='red', label='total dgm')
    ax.set_xlim(m,M)
    ax.set_ylim(m,M)
    ax.add_patch(Polygon([[m, m], [M, m], [M, M]], fill=True, color='lightgrey'))
    ax.set_aspect("equal")
    ax.set_xlabel("Births", fontsize=18)
    ax.set_ylabel("Deaths", fontsize=18)

    ax.legend()
    ax.set_title("Diagrams", fontsize=24)

from datetime import datetime
from tqdm import tqdm

print("hello1")
# Define the projection map
proj = pyproj.Proj(proj='utm', zone=50, ellps='WGS84')
print("hello2")
# Define the DataFrame
df = pd.read_csv("filtered_car_trajectories.csv")

print("hello3")
# Create coordinate projection
x, y = proj(df['Longitude'].tolist(), df['Latitude'].tolist())
p = np.c_[x, y, 0.3048 * df['Altitude']] # convert alt to meters
v = pptk.viewer(p)

print("it still goes well")
# Create persistence diagrams
x_coords = df['Longitude'].tolist()
y_coords = df['Latitude'].tolist()

# Construct the point cloud
print("Constructing point cloud")
point_cloud = np.column_stack((x_coords, y_coords))
print("Computing diagrams")
diagram = ripscomplex(point_cloud[1:10], 2)

print("Plotting diagrams")
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
plot_dgm(diagram, diagram, m =-0.1, M=2, ax=ax1)
