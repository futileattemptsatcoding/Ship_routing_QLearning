# %%
import geopy
import csv
import numpy as np
from geopy.distance import distance
from geopy.geocoders import Nominatim
import time
from math import fabs

# %%
start_coord = [13.0827, 80.2778]
end_coord = [18.9207, 72.8207]
start_coord = [15,70]
min_lat = 5
max_lat = 30
min_lon = 30
max_lon = 100

lat_step = 0.1
lon_step = 0.1

lat_range = np.arange(min_lat, max_lat+1,lat_step)
lon_range = np.arange(min_lon, max_lon+1,lon_step)

reward_weights = {"direction_reward": 100,
                  "wind_speed": 1,
                  "visibility":20,
                  "temp":1,
                  "land_proximity": 10,
                  "land_prox_const":1000,
                  "distance_from_goal":100000,
                  "distance_penalty":50
                  }


# %%
lat_range

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
                if land.contains(point).any():
                    writer.writerow([lat, lon])
                    is_shore = False
                    for ir in [-lat_step,lat_step]:
                        for ic in [-lon_step,lon_step]:
                            if not land.contains(Point(lon+ic, lat+ir)).any():
                                is_shore = True
                                writer_sh.writerow([lat+ir,lon+ic])
                                break
                        if is_shore: break
                    
                lon += lon_step
            lat += lat_step


# %%
import math

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    delta_lon = lon2 - lon1

    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))
    initial_bearing = math.atan2(x, y)

    initial_bearing = math.degrees(initial_bearing)

    bearing = (initial_bearing + 360) % 360

    return bearing


# %%


# %%

with open("LandCoords.csv",'r',newline='') as fh:
    reader = csv.reader(fh)
    with open("ShoreCoords.csv",'r',newline='') as fsh:
        reader_sh = csv.reader(fsh)
        for row1,row2 in zip(reader,reader_sh):
            print(row1,row2)
        

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import csv

shapefile_path = "/Users/krisha/STUFF/env2/NE Admin 0 Countries/ne_110m_admin_0_countries.shp"
land = gpd.read_file(shapefile_path)

fig, ax = plt.subplots(figsize=(12, 8))
land.boundary.plot(ax=ax, linewidth=1, color="black")

with open("LandCoords.csv", 'r') as fh:
    reader = csv.reader(fh)
    land_points = [(float(row[0]), float(row[1])) for row in reader]
    land_lats, land_lons = zip(*land_points)
    ax.scatter(land_lons, land_lats, color="brown", s=10, label="Land Points", alpha=0.7)

with open("ShoreCoords.csv", 'r') as fsh:
    reader_sh = csv.reader(fsh)
    shore_points = [(float(row[0]), float(row[1])) for row in reader_sh]
    shore_lats, shore_lons = zip(*shore_points)
    ax.scatter(shore_lons, shore_lats, color="blue", s=10, label="Shore Points", alpha=0.7)

ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.title("Map of Land and Shore Points")
plt.legend()

plt.show()


# %%
import random

def generate_synthetic_weather(lat, lon):
    temp = random.uniform(15, 40)
    humidity = random.uniform(30, 100)
    wind_speed = random.uniform(0, 15)
    wind_deg = random.uniform(0, 360)
    visibility = random.uniform(1000, 10000)
    cloud = random.uniform(0, 100)
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

def precompute_weather_data(lat_range, lon_range):
    global weather_data 
    weather_data = {}
    
    for lat in lat_range:
        for lon in lon_range:
            weather_data[(lat, lon)] = generate_synthetic_weather(lat, lon)

    weather_data = {f"{lat},{lon}": data for (lat, lon), data in weather_data.items()}
    with open('weather_data.json', 'w') as f:
        json.dump(weather_data, f)

def load_weather_data():
    with open('weather_data.json', 'r') as f:
        weather_data = json.load(f)
    weather_data = {tuple(map(float, key.split(','))): value for key, value in weather_data.items()}
    return weather_data



# %%

optimal_direction = calculate_bearing(start_coord[0], start_coord[1], end_coord[0], end_coord[1])

optimal_direction = calculate_bearing(start_coord[0], start_coord[1], end_coord[0], end_coord[1])

def get_weather_reward(lat, lon, direction):
    current_weather = weather_data.get((lat, lon), generate_synthetic_weather(lat, lon))
    temp = current_weather['main']['temp']
    humidity = current_weather['main']['humidity']
    wind_speed = current_weather['wind']['speed']
    visibility = current_weather['visibility']
    direction_diff = min(abs(direction - optimal_direction), 360 - abs(direction - optimal_direction))
    direction_reward = reward_weights["direction_reward"] * (1 - direction_diff / 180)
    base_reward = (
        reward_weights["wind_speed"] * (10 - wind_speed) +
        (visibility - 1000) * reward_weights["visibility"] +
        reward_weights["temp"] * abs(temp - 30)
    )

    return base_reward + direction_reward


# %%
from rtree import index
from geopy.distance import geodesic

def store_in_rtree():
    idx = index.Index()
    with open("LandCoords.csv",'r') as fh:
        reader = csv.reader(fh)
        for i, (lat,lon) in enumerate(reader):
            idx.insert(i,(float(lon),float(lat),float(lon),float(lat)))
    return idx

# %%
def get_land_reward(idx,lat,lon,end_coord):
    min_dist = float('inf')
    for i in idx.nearest((lon,lat,lon,lat)):
        candidate_coord = reader[i]
        distance = geodesic([lat,lon],candidate_coord).km
        if distance < min_dist:
            min_dist = distance
            nearest_land_coord = candidate_coord
    
    return(reward_weights["land_proximity"]*(reward_weights["land_prox_const"]-min_dist))

def dist_from_goal_reward(lat,lon,next_lat, next_lon, end_coord):
    current_distance = geodesic((lat, lon), (end_coord[0], end_coord[1])).km
    next_distance = geodesic((next_lat, next_lon), (end_coord[0], end_coord[1])).km
    return reward_weights["distance_from_goal"] * (current_distance - next_distance) / current_distance if current_distance > next_distance else -1*(reward_weights["distance_penalty"])  

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
    lat_max = 0
    lon_max = 0
    maxc = 0
    x,y = lat_lon_to_indices(lat,lon)
    for i in range(x-1,x+2):
        for j in range(y-1,y+2):
            if(Q[i][j]>maxc and (i!=x or j!=y)):
                maxc = Q[i][j]
                lat_max = i
                lon_max = j
    return(lat_max,lon_max)

# %%


# %%
global count_threshold, count, del_lat_th, del_lon_th, ref_lat, ref_lon
count_threshold = 15
count = 0
del_lat_th = lat_step
del_lon_th = lon_step
ref_lat = start_coord[0]
ref_lon = start_coord[1]

# %%
def check_if_stuck(lat, lon):
    global ref_lat, ref_lon, count  # Declare the variables as global
    
    if (abs(lat - ref_lat) > del_lat_th) or (abs(lon - ref_lon) > del_lon_th):
        ref_lat = lat
        ref_lon = lon
        count = 0  
    else:
        count += 1 
    
    if count > count_threshold:
        return True
    else:
        return False


# %%
def apply_significant_perturbation(lat,lon):
    goal_lat,goal_lon = end_coord
    lat+= (goal_lat-lat)*lat_step
    lon+= (goal_lon -lon)*lon_step
    return lat,lon

# %%
def propagate_negative_reward(lat_idx,lon_idx,prop_factor=0.5,radius=2):
    for i in range(-radius,radius+1):
        for j in range(-radius,radius+1):
            if i==0 and j==0:
                continue
            new_lat_idx = max(0,min(lat_idx+i,Q.shape[0]-1))
            new_lon_idx = max(0,min(lon_idx+j,Q.shape[1]-1))
            distance = np.sqrt(i**2 + j**2)
            Q[new_lat_idx,new_lon_idx] -= prop_factor*Q[new_lat_idx,new_lon_idx]/distance
            

# %%
import numpy as np
import csv

def initialize_Q_rough():
    n_lat = len(lat_range)+1
    n_lon = len(lon_range)+1
    goal_lat_idx, goal_lon_idx = lat_lon_to_indices(end_coord[0], end_coord[1], lat_step, lon_step)
    n_actions = 8

    global Q
    Q = np.zeros((n_lat, n_lon))

    with open("LandCoords.csv", 'r', newline='') as fh:
        reader = csv.reader(fh)
        for row in reader:
            r, c = lat_lon_to_indices(float(row[0]), float(row[1]), lat_step, lon_step)
            print("Indicies are",r,c)
            Q[r, c] = -1000  

    for lat in lat_range:
        for lon in lon_range:
            lat_idx,lon_idx = lat_lon_to_indices(lat,lon,lat_step,lon_step)
            distance_to_goal = np.sqrt((lat_idx - goal_lat_idx)**2 + (lon_idx - goal_lon_idx)**2)
            Q[lat_idx, lon_idx] = reward_weights["distance_from_goal"] * (1 / (distance_to_goal + 1))

    goal_reward = 100000
    Q[goal_lat_idx, goal_lon_idx] = goal_reward


# %%
def initialize_Q():
    n_lat = len(lat_range) + 1
    n_lon = len(lon_range) + 1
    goal_lat_idx, goal_lon_idx = lat_lon_to_indices(end_coord[0], end_coord[1], lat_step, lon_step)

    global Q
    Q = np.random.uniform(low=0, high=0.1, size=(n_lat, n_lon))  

    with open("LandCoords.csv", 'r', newline='') as fh:
        reader = csv.reader(fh)
        for row in reader:
            r, c = lat_lon_to_indices(float(row[0]), float(row[1]), lat_step, lon_step)
            Q[r, c] = -1000 

    max_distance = np.sqrt((n_lat - 1)**2 + (n_lon - 1)**2)
    for lat_idx in range(n_lat):
        for lon_idx in range(n_lon):
            if Q[lat_idx, lon_idx] >= 0:  
                distance_to_goal = np.sqrt((lat_idx - goal_lat_idx)**2 + (lon_idx - goal_lon_idx)**2)
                Q[lat_idx, lon_idx] += (1 - distance_to_goal / max_distance) * 10  

    boundary_width = 5
    Q[:boundary_width, :] -= 5
    Q[-boundary_width:, :] -= 5
    Q[:, :boundary_width] -= 5
    Q[:, -boundary_width:] -= 5

    goal_reward = 1000
    Q[goal_lat_idx, goal_lon_idx] = goal_reward


    Q = np.clip(Q, -1000, 1000)

# %%
global LAND_FLAG 
LAND_FLAG = -1000

def initialize_Q_linearSpace():
    
    n_lat = len(lat_range) + 1
    n_lon = len(lon_range) + 1
    goal_lat_idx, goal_lon_idx = lat_lon_to_indices(end_coord[0], end_coord[1], lat_step, lon_step)

    global Q
    Q = np.zeros((n_lat, n_lon))

    for lat_idx in range(n_lat):
        for lon_idx in range(n_lon):
            
            current_lat = min_lat + lat_idx * lat_step
            current_lon = min_lon + lon_idx * lon_step
            
           
            distance_to_goal = geodesic(
                (current_lat, current_lon), 
                (end_coord[0], end_coord[1])
            ).kilometers

            
            Q[lat_idx, lon_idx] = 1000 * np.exp(-distance_to_goal/1000) 

    
    with open("LandCoords.csv", 'r', newline='') as fh:
        reader = csv.reader(fh)
        for row in reader:
            r, c = lat_lon_to_indices(float(row[0]), float(row[1]), lat_step, lon_step)
            Q[r, c] = LAND_FLAG  

    
    Q[goal_lat_idx, goal_lon_idx] = 1000

    
    ocean_mask = Q > -1000
    if ocean_mask.any():
        ocean_values = Q[ocean_mask]
        Q[ocean_mask] = np.interp(ocean_values, 
                                 (ocean_values.min(), ocean_values.max()), 
                                 (1, 999))  # Scale ocean values between 1 and 999

# %%
initialize_Q_linearSpace()

# %%
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from shapely.geometry import Point
import matplotlib.patches as patches

# Load country boundaries
shapefile_path = "/Users/krisha/STUFF/env2/NE Admin 0 Countries/ne_110m_admin_0_countries.shp"
land = gpd.read_file(shapefile_path)


# Prepare map for plotting
fig, ax = plt.subplots(figsize=(12, 8))
land.boundary.plot(ax=ax, linewidth=1, color="black")

# Normalize colors for the range of Q values
norm = Normalize(vmin=np.min(Q), vmax=np.max(Q))
cmap = cm.ScalarMappable(norm=norm, cmap="viridis")
cmap.set_array([])

# Plot each cell as a rectangle color-coded by Q matrix value
for i, lat in enumerate(lat_range):
    for j, lon in enumerate(lon_range):
        weight = Q[i, j]
        color = cmap.to_rgba(weight)
        
        # Add a rectangle for each lat-lon cell in the matrix
        ax.add_patch(patches.Rectangle(
            (lon, lat), lon_step, lat_step,
            facecolor=color, edgecolor="none", alpha=0.6
        ))

# Add colorbar and labels
plt.colorbar(cmap, ax=ax, label="Q Matrix Weights")
ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.title("Q Matrix Weighted Map of Region")

plt.show()


Q[-1][-1]

# %%
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import geopandas as gpd

# %%
def q_learning_land(lat_step, lon_step, start_coord, end_coord, alpha=0.1, gamma=0.9, epsilon=0.2, episodes=1,loop_range=12000):
    global Q, actions, lat_range, lon_range
    LAND_FLAG = -1000
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
    num_actions = len(actions)
    num_lat = int((max_lat - min_lat) / lat_step) + 1
    num_lon = int((max_lon - min_lon) / lon_step) + 1
    lat_range = np.arange(min_lat, max_lat + lat_step, lat_step)
    lon_range = np.arange(min_lon, max_lon + lon_step, lon_step)
    
    precompute_weather_data(lat_range, lon_range)
    weather_data = load_weather_data()
    initialize_Q()
    idx = store_in_rtree()
    
    trajectory = []
    
    for episode in range(episodes):
        lat, lon = start_coord[0], start_coord[1]
        trajectory.append((lat, lon))
        count = 0
        
        while geodesic((lat, lon), (end_coord[0], end_coord[1])).km > 1 and count < loop_range:
            current_distance = geodesic((lat, lon), (end_coord[0], end_coord[1])).km
            print(f"Current lat: {lat:.4f}, current lon: {lon:.4f}")
            lat_idx, lon_idx = lat_lon_to_indices(lat, lon, lat_step, lon_step)
            
    
            valid_moves = []
            for action, (dlat, dlon) in enumerate(actions):
                next_lat = lat + dlat * lat_step
                next_lon = lon + dlon * lon_step
                if min_lat <= next_lat <= max_lat and min_lon <= next_lon <= max_lon:
                    next_lat_idx, next_lon_idx = lat_lon_to_indices(next_lat, next_lon, lat_step, lon_step)
                    
                    if Q[next_lat_idx, next_lon_idx] != LAND_FLAG:
                        valid_moves.append((Q[next_lat_idx, next_lon_idx], action))
            
            if not valid_moves:
                print("No valid moves available... applying perturbation")
                lat, lon = apply_significant_perturbation(lat, lon, end_coord)
                propagate_negative_reward(lat_idx, lon_idx)
                continue
            
            if np.random.uniform(0, 1) < epsilon:
                goal_bearing = calculate_bearing(lat, lon, end_coord[0], end_coord[1])
                action_preferences = []
                for q_val, action in valid_moves:
                    dlat, dlon = actions[action]
                    next_lat = lat + dlat * lat_step
                    next_lon = lon + dlon * lon_step
                    action_bearing = calculate_bearing(lat, lon, next_lat, next_lon)
                    bearing_diff = min(abs(action_bearing - goal_bearing), 360 - abs(action_bearing - goal_bearing))
                    action_preferences.append((bearing_diff, action))
                action_preferences.sort()
                action = action_preferences[np.random.randint(0, min(3, len(action_preferences)))][1]
            else:
                # Choose action with highest neighboring Q-value from valid moves
                action = max(valid_moves)[1]
            
           
            next_lat, next_lon = next_state(lat, lon, action, lat_step, lon_step)
            next_lat_idx, next_lon_idx = lat_lon_to_indices(next_lat, next_lon, lat_step, lon_step)
            
            
            direction = calculate_bearing(lat, lon, next_lat, next_lon)
            weather_reward = get_weather_reward(next_lat, next_lon, direction)
            next_distance = geodesic((next_lat, next_lon), (end_coord[0], end_coord[1])).km
            distance_reward = 100 * (current_distance - next_distance) / current_distance
            
            total_reward = weather_reward + distance_reward
            
        
            Q[lat_idx, lon_idx] += alpha * (total_reward + gamma * Q[next_lat_idx, next_lon_idx] - Q[lat_idx, lon_idx])
            
            lat, lon = next_lat, next_lon
            trajectory.append((lat, lon))
            
            if check_if_stuck(lat, lon):
                epsilon = min(1.0, epsilon * 1.5)
                print("Agent is stuck...")
                #apply_significant_perturbation(lat,lon)
            
            count += 1
        
        epsilon = max(0.01, epsilon * 0.99)
    
    plot_trajectory(trajectory, start_coord, end_coord)

# %%
def q_learning_land_perpendicular(lat_step, lon_step, start_coord, end_coord, alpha=0.1, gamma=0.9, epsilon=0.2, episodes=1,loop_range=12000):
    global Q, actions, lat_range, lon_range
    LAND_FLAG = -1000
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
    num_actions = len(actions)
    num_lat = int((max_lat - min_lat) / lat_step) + 1
    num_lon = int((max_lon - min_lon) / lon_step) + 1
    lat_range = np.arange(min_lat, max_lat + lat_step, lat_step)
    lon_range = np.arange(min_lon, max_lon + lon_step, lon_step)
    
    precompute_weather_data(lat_range, lon_range)
    weather_data = load_weather_data()
    initialize_Q()
    idx = store_in_rtree()
    
    trajectory = []
    
    def check_adjacent_land(lat, lon, lat_step, lon_step):
        land_directions = []
        for dlat, dlon in actions:
            next_lat = lat + dlat * lat_step
            next_lon = lon + dlon * lon_step
            if min_lat <= next_lat <= max_lat and min_lon <= next_lon <= max_lon:
                next_lat_idx, next_lon_idx = lat_lon_to_indices(next_lat, next_lon, lat_step, lon_step)
                if Q[next_lat_idx, next_lon_idx] == LAND_FLAG:
                    land_directions.append((dlat, dlon))
        return land_directions

    def calculate_perpendicular_preference(action, land_directions):
        preference = 0
        action_dlat, action_dlon = actions[action]
        
        for land_dlat, land_dlon in land_directions:
            # Calculate dot product - lower means more perpendicular
            dot_product = abs(action_dlat * land_dlat + action_dlon * land_dlon)
            # Convert to preference (higher for more perpendicular movement)
            preference += 1 - dot_product
        
        return preference if land_directions else 0
    
    for episode in range(episodes):
        lat, lon = start_coord[0], start_coord[1]
        trajectory.append((lat, lon))
        count = 0
        
        while geodesic((lat, lon), (end_coord[0], end_coord[1])).km > 1 and count < loop_range:
            current_distance = geodesic((lat, lon), (end_coord[0], end_coord[1])).km
            print(f"Current lat: {lat:.4f}, current lon: {lon:.4f}")
            lat_idx, lon_idx = lat_lon_to_indices(lat, lon, lat_step, lon_step)
            
            # Check for adjacent land
            land_directions = check_adjacent_land(lat, lon, lat_step, lon_step)
            
            valid_moves = []
            for action, (dlat, dlon) in enumerate(actions):
                next_lat = lat + dlat * lat_step
                next_lon = lon + dlon * lon_step
                if min_lat <= next_lat <= max_lat and min_lon <= next_lon <= max_lon:
                    next_lat_idx, next_lon_idx = lat_lon_to_indices(next_lat, next_lon, lat_step, lon_step)
                    if Q[next_lat_idx, next_lon_idx] != LAND_FLAG:
                        perpendicular_bonus = calculate_perpendicular_preference(action, land_directions) * 50  # Adjust multiplier as needed
                        valid_moves.append((Q[next_lat_idx, next_lon_idx] + perpendicular_bonus, action))
            
            if not valid_moves:
                print("No valid moves available... applying perturbation")
                lat, lon = apply_significant_perturbation(lat, lon, end_coord)
                propagate_negative_reward(lat_idx, lon_idx)
                continue
            
            if np.random.uniform(0, 1) < epsilon:
                goal_bearing = calculate_bearing(lat, lon, end_coord[0], end_coord[1])
                action_preferences = []
                for q_val, action in valid_moves:
                    dlat, dlon = actions[action]
                    next_lat = lat + dlat * lat_step
                    next_lon = lon + dlon * lon_step
                    action_bearing = calculate_bearing(lat, lon, next_lat, next_lon)
                    bearing_diff = min(abs(action_bearing - goal_bearing), 360 - abs(action_bearing - goal_bearing))
                    perpendicular_bonus = calculate_perpendicular_preference(action, land_directions) * 30  # Smaller bonus for exploration
                    action_preferences.append((bearing_diff - perpendicular_bonus, action))
                action_preferences.sort()
                action = action_preferences[np.random.randint(0, min(3, len(action_preferences)))][1]
            else:
                action = max(valid_moves)[1]
            
            next_lat, next_lon = next_state(lat, lon, action, lat_step, lon_step)
            next_lat_idx, next_lon_idx = lat_lon_to_indices(next_lat, next_lon, lat_step, lon_step)
            
            direction = calculate_bearing(lat, lon, next_lat, next_lon)
            weather_reward = get_weather_reward(next_lat, next_lon, direction)
            next_distance = geodesic((next_lat, next_lon), (end_coord[0], end_coord[1])).km
            distance_reward = 100 * (current_distance - next_distance) / current_distance
            
            # Add land avoidance reward
            land_avoidance_reward = calculate_perpendicular_preference(action, land_directions) * 40
            total_reward = weather_reward + distance_reward + land_avoidance_reward
            
            Q[lat_idx, lon_idx] += alpha * (total_reward + gamma * Q[next_lat_idx, next_lon_idx] - Q[lat_idx, lon_idx])
            
            lat, lon = next_lat, next_lon
            trajectory.append((lat, lon))
            
            if check_if_stuck(lat, lon):
                epsilon = min(1.0, epsilon * 1.5)
                print("Agent is stuck...")
            
            count += 1
        
        epsilon = max(0.01, epsilon * 0.99)
    
    plot_trajectory(trajectory, start_coord, end_coord)

# %%
def plot_trajectory(trajectory, start_coord, end_coord):
    shapefile_path = "/Users/krisha/STUFF/env2/NE Admin 0 Countries/ne_110m_admin_0_countries.shp"
    land = gpd.read_file(shapefile_path)

    fig, ax = plt.subplots(figsize=(12, 8))
    land.boundary.plot(ax=ax, linewidth=1, color="black")

    # Plot trajectory
    lats, lons = zip(*trajectory)
    ax.plot(lons, lats, 'r-', linewidth=2, alpha=0.7)
    ax.plot(lons, lats, 'bo', markersize=4, alpha=0.5)
    
    # Plot start and end points
    ax.plot(start_coord[1], start_coord[0], 'go', markersize=10, label='Start')
    ax.plot(end_coord[1], end_coord[0], 'ro', markersize=10, label='End')

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.title("Agent Trajectory")
    plt.legend()
    plt.show()

# %%
q_learning_land(lat_step, lon_step, start_coord,end_coord)


