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
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading

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
labels = df['Label'].drop_duplicates()
encodings = df['Encoding'].drop_duplicates()
label_map = dict(zip(labels, encodings))

# Filter the data on latitude and longitude to obtain only the points in centre/suburbs
lat_long_mask = (df['Longitude'] > 115.5) & (df['Longitude'] < 116.7) & (df['Latitude'] > 39.5) & (df['Latitude'] < 40.25)
df = df[lat_long_mask]

# Filter the data on daytime traffic
df['Date_Time'] = pd.to_datetime(df['Date_Time'])
daytime_mask = (df['Date_Time'].dt.hour >= 6) & (df['Date_Time'].dt.hour < 10)  # from 06.00 to 10.00
df = df[daytime_mask]

# Filter the data on car traffic
car_encoding = label_map['car']
car_mask = (df['Encoding'] == car_encoding)
df = df[car_mask]

df['Long_meters'] = to_meters(df['Longitude'].values)
df['Lat_meters'] = to_meters(df['Latitude'].values)

df_new = pd.DataFrame(columns=['Long_meters', 'Lat_meters', 'Altitude', 'Longitude', 'Latitude'])


# Check if a point is eligible to be added to df_new
def is_eligible(row, queue):
    for new_row in queue:
        if np.abs(new_row[0] - row['Long_meters']) < 50 and np.abs(new_row[1] - row['Lat_meters']) < 50:
            return False
    return True


def process_row(row_tuple, queue, lock):
    _, row = row_tuple
    with lock:
        if is_eligible(row, queue.queue):
            result = [row['Long_meters'], row['Lat_meters'], row['Altitude'], row['Longitude'], row['Latitude']]
            queue.put(result)
            return result
    return None


# Filter the data on points that are close to each other
df_new_queue = Queue()
eligibility_lock = threading.Lock()
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_row, row_tuple, df_new_queue, eligibility_lock) for row_tuple in df.iterrows()]
    results = []
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        if result is not None:
            results.append(result)

for result in tqdm(results, total=len(results)):
    if result is not None:
        df_new.loc[len(df_new)] = result

# Visualize the filtered points
x, y = proj(df['Longitude'].tolist(), df['Latitude'].tolist())
p = np.c_[x, y, 0.3048 * df['Altitude']]  # convert alt to meters
v = pptk.viewer(p)

df = df_new
df.to_csv('filtered_taxi_trajectories.csv', index=False)

print(df.head(100))

# Visualize the filtered points
x, y = proj(df['Longitude'].tolist(), df['Latitude'].tolist())
p = np.c_[x, y, 0.3048 * df['Altitude']]  # convert alt to meters
v = pptk.viewer(p)
