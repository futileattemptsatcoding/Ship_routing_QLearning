import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

start_coord = [17.6833, 83.2167]
end_coord = [13.0827, 80.2778]


def plot_explored_coords(csv_path, shapefile_path, start_coord, end_coord):
    land = gpd.read_file(shapefile_path)
    
    data = pd.read_csv(csv_path)
    lats = data['Latitude']
    lons = data['Longitude']
    F_values = data['F-Value'] 


    fig, ax = plt.subplots(figsize=(12, 8))
    land.boundary.plot(ax=ax, linewidth=1, color="black") 

    scatter = ax.scatter(lons, lats, c=F_values, cmap='viridis', s=10, label='Explored', alpha=0.7)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('F-Value Intensity')

    ax.plot(start_coord[1], start_coord[0], 'go', markersize=10, label='Start')
    ax.plot(end_coord[1], end_coord[0], 'ro', markersize=10, label='End')

    ax.set_xlim(30, 100)
    ax.set_ylim(5, 30)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.title("Explored Coordinates with A* Algorithm")
    plt.legend()
    plt.show()


def plot_route(csv_path, shapefile_path, start_coord, end_coord):
    land = gpd.read_file(shapefile_path)
    
    data = pd.read_csv(csv_path)
    lats = data['Latitude']
    lons = data['Longitude']

    fig, ax = plt.subplots(figsize=(12, 8))
    land.boundary.plot(ax=ax, linewidth=1, color="black")  \

    ax.scatter(lons, lats, color='blue', s=10, label='Explored', alpha=0.7)

    ax.plot(start_coord[1], start_coord[0], 'go', markersize=10, label='Start')
    ax.plot(end_coord[1], end_coord[0], 'ro', markersize=10, label='End')

    ax.set_xlim(30, 100)
    ax.set_ylim(5, 30)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.title("Explored Coordinates with A* Algorithm")
    plt.legend()
    plt.show()
csv_path = "explored.csv"  
shapefile_path = "NE Admin 0 Countries/ne_110m_admin_0_countries.shp"
route = "AStarShipRoute/route.csv"


plot_explored_coords(csv_path, shapefile_path, start_coord, end_coord)
#plot_route(route, shapefile_path,start_coord,end_coord)
