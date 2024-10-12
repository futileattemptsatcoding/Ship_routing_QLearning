# %%
import geopy
import csv
import numpy as np

# %%
# start_coord = [18.9207, 72.8207] #latitude, longitude
# end_coord = [13.0827, 80.2778]

start_coord = [13.0827, 80.2778]
end_coord = [18.9207, 72.8207]


# %%
from geopy.distance import distance
from geopy.geocoders import Nominatim
import time
from math import fabs

# %%
min_lat = 5
max_lat = 30
min_lon = 30
max_lon = 100

lat_step = 0.1
lon_step = 0.1


# %%
import geopandas as gpd
from shapely.geometry import Point
import csv

# Load the shapefile from the downloaded data
shapefile_path = "/Users/krisha/STUFF/env2/NE Admin 0 Countries/ne_110m_admin_0_countries.shp"
land = gpd.read_file(shapefile_path)

with open('LandCoords.csv', 'w', newline='') as fh:
    writer = csv.writer(fh)
    with open('ShoreCoords.csv','w',newline='') as fsh:
        writer_sh = csv.writer(fsh)
        lat = min_lat
        while lat <= max_lat:
            lon = min_lon
            while lon <= max_lon:
                point = Point(lon, lat)
                # Check if the point is within any land area
                if land.contains(point).any():
                    writer.writerow([lat, lon])
                    #check if theres a nearby water area so that it becomes a shore.
                    flag = False
                    for ir in [-1,1]:
                        for ic in [-1,1]:
                            if not land.contains(Point(lon+ic, lat+ir)).any():
                                flag = True
                                writer_sh.writerow([lat+ir,lon+ic])
                                break
                    
                lon += lon_step
            lat += lat_step


# %%

with open("LandCoords.csv",'r',newline='') as fh:
    reader = csv.reader(fh)
    with open("ShoreCoords.csv",'r',newline='') as fsh:
        reader_sh = csv.reader(fsh)
        for row1,row2 in zip(reader,reader_sh):
            print(row1,row2)
        

# %%
import requests
API_KEY = '' #use your own key, people

# %%
import random

def generate_synthetic_weather(lat, lon):
    # Simulating temperature (range 15°C to 40°C)
    temp = random.uniform(15, 40)
    
    # Simulating humidity (range 30% to 100%)
    humidity = random.uniform(30, 100)
    
    # Simulating wind speed (range 0 to 15 m/s)
    wind_speed = random.uniform(0, 15)
    
    # Simulating wind direction (0 to 360 degrees)
    wind_deg = random.uniform(0, 360)
    
    # Simulating visibility (in meters, range 1000m to 10000m)
    visibility = random.uniform(1000, 10000)
    
    # Simulating cloud cover (percentage, 0% to 100%)
    cloud = random.uniform(0, 100)
    
    # Pack the data into a dictionary to simulate the structure of real weather data
    weather_data = {
        'main': {
            'temp': temp,
            'humidity': humidity
        },
        'wind': {
            'speed': wind_speed,
            'deg': wind_deg
        },
        'visibility': visibility,
        'clouds': {
            'all': cloud
        }
    }
    
    return weather_data


# %%
import json

# Function to generate and store weather data for all grid points
def precompute_weather_data(lat_range, lon_range, lat_step, lon_step):
    weather_data = {}
    for lat in lat_range:
        for lon in lon_range:
            weather_data[(lat, lon)] = generate_synthetic_weather(lat, lon)
    # Convert the dictionary to a format suitable for JSON
    weather_data = {f"{lat},{lon}": data for (lat, lon), data in weather_data.items()}
    with open('weather_data.json', 'w') as f:
        json.dump(weather_data, f)

# Function to load precomputed weather data
def load_weather_data():
    with open('weather_data.json', 'r') as f:
        weather_data = json.load(f)
    # Convert back the dictionary to the original format
    weather_data = {tuple(map(float, key.split(','))): value for key, value in weather_data.items()}
    return weather_data

precompute_weather_data(lat_range, lon_range, lat_step, lon_step)
weather_data = load_weather_data()

# %%
def get_weather_reward(lat,lon,direction):
    # current_weather_url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric'
    # response = requests.get(current_weather_url)
    # current_weather = response.json()
    # print(current_weather)
    # print(lat,lon)
    # #5 day forecast
    # forecast_url = f'http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric'
    # response = requests.get(forecast_url)
    # forecast_data = response.json()
    
    #for now use simulated data:
    current_weather = weather_data.get((lat, lon), generate_synthetic_weather(lat, lon))
    temp = current_weather['main']['temp']
    humidity = current_weather['main']['humidity']
    wind_speed = current_weather['wind']['speed']
    wind_deg = current_weather['wind']['deg']
    cloud = current_weather['clouds']['all']
    visibility = current_weather['visibility']
    reward = (10 - wind_speed) + (visibility - 1000) * 20 + fabs(temp - 30)
    ##reward += direction_alignment_reward
    return reward

# %%
from rtree import index
from geopy.distance import geodesic

def store_in_rtree():
    idx = index.Index()
    with open('LandCoords.csv','r') as fh:
        reader = csv.reader(fh)
        for i, (lat,lon) in enumerate(reader):
            idx.insert(i,(float(lon),float(lat),float(lon),float(lat)))
    return idx

with open('LandCoords.csv','r') as fh:
        reader = list(csv.reader(fh))

def get_land_proximity_reward(idx,lat,lon,end_coord):
    min_dist = float('inf')
    for i in idx.nearest((lon,lat,lon,lat)):
        candidate_coord = reader[i]
        distance = geodesic([lat,lon],candidate_coord).km
        if distance < min_dist:
            min_dist = distance
            nearest_land_coord = candidate_coord
    
    return(-10*min_dist - 10*geodesic([lat,lon],[end_coord[0],end_coord[1]]))


# %%
#populating the reward matrix:
R = [[]]
# row = 0
# col = 0
# for i, lat in enumerate(np.arange(min_lat,max_lat,lat_step)):
#     for j, lon in enumerate(np.arange(min_lon,max_lon,lon_step)):
#         weather_reward = get_weather_reward(i,j)
#         land_reward = get_land_proximity_reward(i,j)
#         R[i][j] = weather_reward + land_reward


# %%
num_states = len(np.arange(min_lat, max_lat, lat_step)) * len(np.arange(min_lon, max_lon, lon_step))
num_actions = 8  # Possible directions: N, S, E, W, NE, NW, SE, SW
lat_range = np.arange(min_lat, max_lat, lat_step)
lon_range = np.arange(min_lon, max_lon, lon_step)
num_lat = len(lat_range)
num_lon = len(lon_range)

#alt: only look at the recatangle with diagonal points start and end
#coordinates? Will reduce complexity but might give more inefficient result.

# %%

def next_state(lat, lon, action,lat_step,lon_step):
    lt, ln = actions[action]
    return lat + lat_step*lt, lon + lon_step*ln


# %%
def lat_lon_to_indices(lat, lon,lat_step,lon_step):
    lat_idx = int((lat - min_lat) / lat_step)
    lon_idx = int((lon - min_lon) / lon_step)
    return(lat_idx, lon_idx)


# %%
def check_for_max(lat,lon):
    max_lat = 0
    max_lon = 0
    maxc = 0
    x,y = lat_lon_to_indices(lat,lon)
    for i in range(x-1,x+2):
        for j in range(y-1,y+2):
            if(Q[i][j]>maxc or (i!=x or j!=y)):
                maxc = Q[i][j]
                max_lat = i
                max_lon = j
    return(max_lat,max_lon)

# %%
import math

def calculate_bearing(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Difference in longitudes
    delta_lon = lon2 - lon1

    # Bearing calculation
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))
    initial_bearing = math.atan2(x, y)

    # Convert bearing from radians to degrees
    initial_bearing = math.degrees(initial_bearing)

    # Normalize the bearing to be within 0-360 degrees
    bearing = (initial_bearing + 360) % 360

    return bearing


# %%
global optimal_direction, count_threshold, count, del_lat_th, del_lon_th, ref_lat, ref_lon
optimal_direction = calculate_bearing(start_coord[0], start_coord[1], end_coord[0], end_coord[1])
count_threshold = 15
count = 0
del_lat_th = 0.01
del_lon_th = 0.01
ref_lat = start_coord[0]
ref_lon = start_coord[1]

# %%
def check_if_stuck(lat, lon):
    global ref_lat, ref_lon, count  # Declare the variables as global
    
    if (abs(lat - ref_lat) > del_lat_th) or (abs(lon - ref_lon) > del_lon_th):
        ref_lat = lat
        ref_lon = lon
        count = 0  # Reset the count if significant movement is detected
    else:
        count += 1  # Increment count if the point hasn't moved significantly
    
    if count > count_threshold:
        return True
    else:
        return False


# %%
def apply_random_perturbation(current_lat, current_lon,lat_op,lon_op):
    perturb_lat = current_lat + lat_op*random.uniform(0, 0.1)
    perturb_lon = current_lon + lon_op*random.uniform(0, 0.1)  
    return perturb_lat, perturb_lon

def apply_significant_perturbation_1(lat, lon, lat_op, lon_op, end_coord):
    # Move towards the goal with some randomness
    goal_lat, goal_lon = end_coord
    lat += lat_step*int(lat_op*(goal_lat - lat) *random.uniform(0.1, 0.3))
    lon += lon_step*int(lon_op*(goal_lon - lon) *random.uniform(0.1, 0.3))
    return lat,lon

def apply_significant_perturbation(lat, lon, lat_op, lon_op, end_coord):
    # Move towards the goal with some randomness
    goal_lat, goal_lon = end_coord
    lat += lat_op*(goal_lat - lat)*0.1
    lon += lon_op*(goal_lon - lon)*0.1
    return lat,lon
    
    

# %%
def propagate_negative_reward(lat_idx,lon_idx,action,propagation_factor=0.5,radius=2):
    for i in range(-radius,radius+1):
        for j in range(-radius,radius+1):
            if i==0 and j==0:
                continue
            new_lat_idx = max(0,min(lat_idx+i,Q.shape[0]-1))
            new_lon_idx = max(0,min(lon_idx+j,Q.shape[1]-1))
            distance = np.sqrt(i**2 + j**2)
            Q[new_lat_idx,new_lon_idx,action] -= propagation_factor*Q[lat_idx,lon_idx,action]/distance
            

# %%
#initializing Q matrix:
def initialize_Q_matrix(lat_step,lon_step):
    with open("LandCoords.csv",'r',newline='') as fh:
        reader = csv.reader(fh)
        for row in reader:
            r,c = lat_lon_to_indices(float(row[0]),float(row[1]),lat_step,lon_step)
            Q[r,c] = -100000
    
    with open("ShoreCoords.csv",'r',newline='') as fsh:
        reader = csv.reader(fsh)
        for row in reader:
            r,c = lat_lon_to_indices(float(row[0]),float(row[1]),lat_step,lon_step)
            Q[r,c] = 100
    
    r,c = lat_lon_to_indices(end_coord[0],end_coord[1],lat_step,lon_step)
    Q[r,c] = 100000
    print("Q matrix initialized")
    

# %%
from geopy.distance import geodesic

def q_learning(lat_step,lon_step,start_coord, end_coord, alpha=0.1, gamma=0.9, epsilon=0.2, episodes=1):
    global Q, actions, lat_range, lon_range
    actions = [
    (-1, 0),  # South
    (1, 0),   # North
    (0, -1),  # West
    (0, 1),   # East
    (-1, -1), # Southwest
    (-1, 1),  # Southeast
    (1, -1),  # Northwest
    (1, 1)    # Northeast
    ] 
    num_actions=len(actions)
    num_lat = int((max_lat - min_lat)*lat_step) +1
    num_lon = int((max_lon - min_lon)*lon_step) +1
    lat_range = np.arange(min_lat,max_lat+1,lat_step)
    lon_range = np.arange(min_lon,max_lon+1,lon_step)
    initialize_Q_matrix(lat_step,lon_step)
    idx = store_in_rtree()
    
    for episode in range(episodes):
        lat, lon = start_coord[0], start_coord[1]
        prev_lat, prev_lon = lat, lon
        count  = 0
        while geodesic((lat, lon), (end_coord[0], end_coord[1])).km > 1 and count<3500:  # Stop when close to destination (e.g., 1 km)
            print("Current lat: ", lat, "current lon: ", lon, " ", end='')
            lat_idx, lon_idx = lat_lon_to_indices(lat, lon,lat_step,lon_step)
            
            # Calculate rewards
            direction = calculate_bearing(prev_lat, prev_lon, lat, lon)
            weather_reward = get_weather_reward(lat, lon, direction)
            #land_reward = get_land_proximity_reward(idx, lat, lon,end_coord)
            reward = weather_reward #+ land_reward
            print("reward: ",reward)
            
            # Markov selection process
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, num_actions)
            else:
                action = np.argmax(Q[lat_idx, lon_idx])  # Choose the action with the maximum Q-value
            
            # If close to the destination, stop random moves and go straight to the goal
            if geodesic((lat, lon), (end_coord[0], end_coord[1])).km < 5:  # "Close enough" condition
                action = np.argmin([
                    geodesic(next_state(lat, lon, a, lat_step, lon_step), end_coord).km
                    for a in range(num_actions)
                ])
            
            # Calculate next state
            next_lat, next_lon = next_state(lat, lon, action, lat_step, lon_step)
            
            # Check boundaries
            if min_lat <= next_lat <= max_lat and min_lon <= next_lon <= max_lon:
                next_lat_idx, next_lon_idx = lat_lon_to_indices(next_lat, next_lon,lat_step,lon_step)
                next_state_valid = True
            else:
                next_state_valid = False
                # if min_lat >= next_lat:
                #     lat_op = 1
                # elif max_lat <= next_lat:
                #     lat_op = -1
                # else:
                #     lat_op = 0.3

                # if min_lon >= next_lon:
                #     lon_op = 1
                # elif max_lon <= next_lon:
                #     lon_op = -1
                # else:
                #     lon_op = 0.3
                Q[lat_idx, lon_idx, action] += alpha * (-100 - Q[lat_idx, lon_idx, action])
                lat,lon = apply_significant_perturbation_1(lat,lon,1,1,end_coord)
                #propagating negative reward across nearby states:
                propagate_negative_reward(lat_idx,lon_idx,action)
                print("outer boundary... point modified...")
            
            if next_state_valid:
                # Q-learning update
                next_max_q = np.max(Q[next_lat_idx, next_lon_idx])
                Q[lat_idx, lon_idx, action] += alpha * (reward + gamma * next_max_q - Q[lat_idx, lon_idx, action])
                
                # Move to the next state
                prev_lat, prev_lon = lat, lon
                lat, lon = next_lat, next_lon
            
            
            #check if its stuck
            if check_if_stuck(lat,lon):
                epsilon *= 1.5
                print("Hey its stuck...")
            count += 1

            
        epsilon = max(0.01, epsilon * 0.99)


# %%
# start_coord = [13.0827, 80.2778]
# end_coord = [18.9207, 72.8207]
start_coord = [13, 80]
end_coord = [19, 73]
lat_step = 1
lon_step = 1
q_learning(lat_step,lon_step,start_coord,end_coord,epsilon=0.2)
print("done")

# %%
def find_nearest_state(lat, lon):
    lat_idx = int(round((lat - min_lat) / lat_step))
    lon_idx = int(round((lon - min_lon) / lon_step))
    return max(0, min(lat_idx, num_lat - 1)), max(0, min(lon_idx, num_lon - 1))

def navigate_to_goal(start_coord, end_coord, max_steps=1000):
    lat, lon = start_coord
    path = [(lat, lon)]
    
    for step in range(max_steps):
        if geodesic((lat, lon), end_coord).km <= 1:
            print(f"Goal reached in {step} steps!")
            break
        
        lat_idx, lon_idx = find_nearest_state(lat, lon)
        action = np.argmax(Q[lat_idx, lon_idx])
        lat, lon = next_state(lat, lon, action, lat_step, lon_step)
        path.append((lat, lon))
        
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            lat, lon = apply_significant_perturbation(lat, lon, end_coord)
            path.append((lat, lon))
    
    return path

# %%
# path = navigate_to_goal(start_coord,end_coord)
# for i in path:
#     print(i)
len(Q)

# %% [markdown]
# TO implement:
# 1. Reduce the grid size from (num_lat)$\times$(num_lon) to step wise grid formation (controllable precision).
# 2. Apply significant perturbation 1 may be faster but random values wont align with grid levels. To caluclate (int) $\times$ (step value)$\times$(operator)

# %%
# import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap

# plt.figure(figsize=(10,8))
# m = Basemap(projection='merc',llcrnrlat=min_lat,urcrnrlat=max_lat,llcrnrlon=min_lon,urcrnrlon=max_lon,resolution='c')
# m.drawcoastlines()
# m.drawcountries()
# m.drawmapboundary()

# for lon,lat in path:
#     x,y = m(lon,lat)
#     m.plot(x,y,'bo',markersize=5)

# #connecting points with lines
# lons,lats = zip(*path)
# x,y = m(lons,lats)
# m.plot(x,y,'r-',markersize=5)

# plt.title("map")
# plt.show()

# %% [markdown]
# 


