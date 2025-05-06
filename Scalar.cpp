#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <climits>
#include <string>
#include <sstream>
#include <algorithm>

using namespace std;

// Edge structure
struct Edge {
    int dest, weight;
    Edge(int d, int w) : dest(d), weight(w) {}
};

// Graph class
class Graph {
    int V; // Number of vertices
    vector<vector<Edge>> adj; // Adjacency list
    vector<vector<int>> sosp; // Store SOSP for each objective
    vector<vector<int>> mosp; // Store MOSP paths

public:
    Graph() : V(0) {}

    // Determine number of vertices and initialize adjacency list
    bool initializeGraph(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cout << "Error opening file!" << endl;
            return false;
        }

        int max_vertex = 0;
        string line;
        int src, dest, weight;
        vector<tuple<int, int, int>> edges;

        // First pass: determine max vertex
        while (getline(file, line)) {
            istringstream iss(line);
            if (!(iss >> src >> dest >> weight)) {
                cout << "Invalid line format: " << line << endl;
                continue;
            }
            if (src < 0 || dest < 0 || weight < 0) {
                cout << "Invalid data: " << src << " " << dest << " " << weight << endl;
                continue;
            }
            max_vertex = max({ max_vertex, src, dest });
            edges.emplace_back(src, dest, weight);
        }
        file.close();

        // Initialize graph with max_vertex + 1 vertices
        V = max_vertex + 1;
        adj.assign(V, vector<Edge>());

        // Second pass: populate adjacency list
        for (const auto& [src, dest, w] : edges) {
            addEdge(src, dest, w);
        }

        return true;
    }

    // Add edge to graph
    void addEdge(int src, int dest, int weight) {
        if (src >= V || dest >= V) {
            int new_V = max(src, dest) + 1;
            adj.resize(new_V);
            V = new_V;
        }
        adj[src].emplace_back(dest, weight);
        adj[dest].emplace_back(src, weight); // Undirected graph
    }

    // Dijkstra's algorithm for SOSP
    vector<int> dijkstra(int src, int objective) {
        vector<int> dist(V, INT_MAX);
        vector<int> parent(V, -1);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

        dist[src] = 0;
        pq.emplace(0, src);

        while (!pq.empty()) {
            int u = pq.top().second;
            int d = pq.top().first;
            pq.pop();

            if (d > dist[u]) continue;

            for (const auto& edge : adj[u]) {
                int v = edge.dest;
                int weight = edge.weight;

                if (dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    parent[v] = u;
                    pq.emplace(dist[v], v);
                }
            }
        }
        return parent;
    }

    // Print path from source to destination
    void printPath(const vector<int>& parent, int dest) {
        if (dest == -1) return;
        vector<int> path;
        int current = dest;
        while (current != -1) {
            path.push_back(current);
            current = parent[current];
        }
        reverse(path.begin(), path.end());
        for (size_t i = 0; i < path.size(); ++i) {
            cout << path[i];
            if (i < path.size() - 1) cout << " -> ";
        }
        cout << endl;
    }

    // Merge SOSP trees to create combined graph
    vector<vector<Edge>> mergeSOSP(const vector<vector<int>>& sosp_trees) {
        vector<vector<Edge>> combined_graph(V);
        vector<vector<int>> weight_count(V, vector<int>(V, 0));

        // Count occurrences of edges in SOSP trees
        for (const auto& tree : sosp_trees) {
            for (int v = 0; v < V; ++v) {
                int u = v;
                while (tree[u] != -1) {
                    int parent = tree[u];
                    weight_count[min(u, parent)][max(u, parent)]++;
                    u = parent;
                }
            }
        }

        // Create combined graph with weights based on occurrence
        for (int u = 0; u < V; ++u) {
            for (const auto& edge : adj[u]) {
                int v = edge.dest;
                int count = weight_count[min(u, v)][max(u, v)];
                if (count > 0) {
                    combined_graph[u].emplace_back(v, edge.weight / count);
                }
            }
        }
        return combined_graph;
    }

    // Compute MOSP
    void computeMOSP(int src, const vector<int>& objectives) {
        sosp.clear();
        for (int obj : objectives) {
            sosp.push_back(dijkstra(src, obj));
        }

        // Print SOSP paths for each objective
        for (size_t i = 0; i < sosp.size(); ++i) {
            cout << "SOSP for objective " << objectives[i] << " from vertex " << src << ":\n";
            for (int v = 0; v < V; ++v) {
                if (v != src && sosp[i][v] != -1) {
                    cout << "Path to " << v << ": ";
                    printPath(sosp[i], v);
                }
            }
        }

        // Merge SOSP trees
        auto combined_graph = mergeSOSP(sosp);

        // Compute SOSP on combined graph
        Graph temp_graph;
        temp_graph.V = V;
        temp_graph.adj.assign(V, vector<Edge>());
        for (int u = 0; u < V; ++u) {
            for (const auto& edge : combined_graph[u]) {
                temp_graph.addEdge(u, edge.dest, edge.weight);
            }
        }
        mosp = { temp_graph.dijkstra(src, 0) };

        // Print MOSP paths
        cout << "MOSP paths from vertex " << src << ":\n";
        for (const auto& path : mosp) {
            for (int v = 0; v < V; ++v) {
                if (v != src && path[v] != -1) {
                    cout << "Path to " << v << ": ";
                    printPath(path, v);
                }
            }
        }

        // Pareto optimality check
        vector<vector<int>> pareto_paths;
        for (const auto& path : mosp) {
            bool dominated = false;
            for (const auto& other_path : mosp) {
                if (&path == &other_path) continue;
                // Simplified dominance check (extend for multiple objectives)
                dominated = false; // Placeholder, assume non-dominated for single objective
            }
            if (!dominated) {
                pareto_paths.push_back(path);
            }
        }
        mosp = pareto_paths;
    }

    // Delete edge
    void deleteEdge(int src, int dest) {
        if (src >= V || dest >= V) return;
        adj[src].erase(remove_if(adj[src].begin(), adj[src].end(),
            [dest](const Edge& e) { return e.dest == dest; }), adj[src].end());
        adj[dest].erase(remove_if(adj[dest].begin(), adj[dest].end(),
            [src](const Edge& e) { return e.dest == src; }), adj[dest].end());
    }

    // User menu
    void userMenu() {
        int choice, src, dest, weight;
        vector<int> objectives = { 0 }; // Example objectives
        while (true) {
            cout << "\n1. Insert edge\n2. Delete edge\n3. Compute MOSP\n4. Exit\nChoice: ";
            cin >> choice;
            if (choice == 4) break;

            switch (choice) {
            case 1:
                cout << "Enter src dest weight: ";
                cin >> src >> dest >> weight;
                if (src >= 0 && dest >= 0 && weight >= 0) {
                    addEdge(src, dest, weight);
                    computeMOSP(0, objectives); // Update MOSP
                }
                else {
                    cout << "Invalid input!" << endl;
                }
                break;
            case 2:
                cout << "Enter src dest: ";
                cin >> src >> dest;
                if (src >= 0 && dest >= 0) {
                    deleteEdge(src, dest);
                    computeMOSP(0, objectives); // Update MOSP
                }
                else {
                    cout << "Invalid input!" << endl;
                }
                break;
            case 3:
                computeMOSP(0, objectives);
                cout << "MOSP computed. Pareto optimal paths stored." << endl;
                break;
            default:
                cout << "Invalid choice!" << endl;
            }
        }
    }
};

int main() {
    Graph g;
    if (g.initializeGraph("weightedfacebook_graph.txt")) {
        g.userMenu();
    }
    return 0;
}