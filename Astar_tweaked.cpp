#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib> 
#include <ctime>   
#include <queue>
#include <limits>

using namespace std;

const double min_lat = 5.0;
const double max_lat = 30.0;
const double min_lon = 30.0;
const double max_lon = 100.0;
const double lat_step = 0.1;
const double lon_step = 0.1;
const string fname = "LandCoords.csv";
const double EARTH_RADIUS_KM = 6371.0;
pair<double, double> End = {13.0827, 80.2778};
pair<double, double> Start = {17.6833, 83.2167};

const int num_rows = static_cast<int>((max_lat - min_lat) / lat_step) + 1;
const int num_cols = static_cast<int>((max_lon - min_lon) / lon_step) + 1;

struct Node {
    double lat, lon;
    bool is_obstacle = false;
    double G = numeric_limits<double>::infinity();
    double H = numeric_limits<double>::infinity();
    double F = numeric_limits<double>::infinity();
    Node* last = nullptr;
    float weather_reward;

    Node(double latitude, double longitude, float weather)
        : lat(latitude), lon(longitude), weather_reward(weather) {}
};

Node* grid[num_rows][num_cols];

struct Compare {
    bool operator()(const pair<Node*, double>& p1, const pair<Node*, double>& p2) {
        return (p1.second > p2.second);
    }
};

pair<int, int> latLonToIndices(double lat, double lon) {
    int lat_idx = static_cast<int>((lat - min_lat) / lat_step);
    int lon_idx = static_cast<int>((lon - min_lon) / lon_step);
    return {lat_idx, lon_idx};
}

pair<double, double> indicesToLatLon(int lat_idx, int lon_idx) {
    double lat = min_lat + lat_idx * lat_step;
    double lon = min_lon + lon_idx * lon_step;
    return {lat, lon};
}

void markObstacles() {
    ifstream file(fname);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        double lat, lon;
        char comma;
        if (ss >> lat >> comma >> lon) {
            pair<int, int> p = latLonToIndices(lat, lon);
            if (p.first >= 0 && p.first < num_rows && p.second >= 0 && p.second < num_cols) {
                grid[p.first][p.second]->is_obstacle = true;
            }
        }
    }
    file.close();
}

double toRadians(double degrees) {
    return degrees * M_PI / 180.0;
}

double geodesic(double lat1, double lon1, double lat2, double lon2) {
    lat1 = toRadians(lat1);
    lon1 = toRadians(lon1);
    lat2 = toRadians(lat2);
    lon2 = toRadians(lon2);

    double dLat = lat2 - lat1;
    double dLon = lon2 - lon1;
    double a = sin(dLat / 2) * sin(dLat / 2) +
               cos(lat1) * cos(lat2) * sin(dLon / 2) * sin(dLon / 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    return EARTH_RADIUS_KM * c;
}

void initializeGrid() {
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            double lat = min_lat + i * lat_step;
            double lon = min_lon + j * lon_step;
            double baseReward = 10.0;

            if (lat > 15.0 && lat < 25.0) {
                baseReward -= 5.0;
            } else if (lat <= 15.0) {
                baseReward -= 10.0;
            }

            double randomVariation = (rand() % 11 - 5) / 2.0;
            grid[i][j] = new Node(lat, lon, baseReward + randomVariation);
        }
    }
}

void AStar_without_weather() {
    vector<pair<double, double>> route;
    priority_queue<pair<Node*, double>, vector<pair<Node*, double>>, Compare> open;
    ofstream explored_file("explored.csv");
    explored_file << "Latitude,Longitude,F-Value,G-Value,H-Value\n";
    Node* start_node = grid[latLonToIndices(Start.first, Start.second).first][latLonToIndices(Start.first, Start.second).second];
    Node* end_node = grid[latLonToIndices(End.first, End.second).first][latLonToIndices(End.first, End.second).second];
    start_node->G = 0.0;
    start_node->H = fabs(geodesic(Start.first, Start.second, End.first, End.second));
    start_node->F = start_node->G + start_node->H;
    open.push(make_pair(start_node, start_node->F));

    Node* curNode = nullptr;
    while (!open.empty()) {
        curNode = open.top().first;
        open.pop();
         explored_file << curNode->lat << "," 
                      << curNode->lon << "," 
                      << curNode->F << "," 
                      << curNode->G << "," 
                      << curNode->H << "\n";
                      

        if (curNode == end_node) break;

        pair<int, int> curIdx = latLonToIndices(curNode->lat, curNode->lon);

        for (int i = 2; i >= -1; i--) {
            for (int j = 2; j >= -1; j--) {
                if ((i == 0 && j == 0) ) continue;
                int ni = curIdx.first + i;
                int nj = curIdx.second + j;

                if (ni < 0 || ni >= num_rows || nj < 0 || nj >= num_cols) continue;
                Node* nbrNode = grid[ni][nj];

                if (nbrNode->is_obstacle) continue;

                double newG = curNode->G + fabs(geodesic(curNode->lat, curNode->lon, nbrNode->lat, nbrNode->lon));
                if (newG < nbrNode->G) {
                    nbrNode->G = newG;
                    nbrNode->H = fabs(geodesic(nbrNode->lat, nbrNode->lon, end_node->lat, end_node->lon));
                    nbrNode->F = nbrNode->G + 10*nbrNode->H;
                    nbrNode->last = curNode;
                    if(!nbrNode->is_obstacle) open.push(make_pair(nbrNode, nbrNode->F));
                }
            }
        }
    }

    explored_file.close();
    cout << "modified explored file" << endl;


    if (curNode == end_node) {
        ofstream csv_file("route.csv");
        csv_file << "Latitude,Longitude\n";

        Node* backtrackNode = end_node;
        while (backtrackNode != nullptr) {
            csv_file << backtrackNode->lat << "," << backtrackNode->lon << "\n";
            backtrackNode = backtrackNode->last;
        }

        csv_file.close();
    }
}

int main() {
    cout << "Starting..." << endl;
    initializeGrid();
    cout << "Grid initialized." << endl;
    markObstacles();
    cout << "Obstacles marked." << endl;    
    AStar_without_weather();
    cout << "Pathfinding complete." << endl;
    return 0;
}
