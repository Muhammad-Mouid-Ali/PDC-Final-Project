#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <algorithm>
#include <chrono>
#include <random>

// Custom hash for pair<int, int>
struct pair_hash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

// Structure to represent an edge
struct Edge {
    int src, dest;
    std::vector<int> weights;

    Edge() : src(0), dest(0), weights() {}
    Edge(int s, int d, const std::vector<int>& w)
        : src(s), dest(d), weights(w) {}
};

// Structure to represent distance for multiple objectives
struct Distance {
    std::vector<int> values;
    
    Distance(int num_objectives, int value = std::numeric_limits<int>::max()) {
        values.resize(num_objectives, value);
    }
    
    bool dominates(const Distance& other) const {
        bool at_least_one_better = false;
        for (size_t i = 0; i < values.size(); i++) {
            if (values[i] > other.values[i]) {
                return false;
            }
            if (values[i] < other.values[i]) {
                at_least_one_better = true;
            }
        }
        return at_least_one_better;
    }
};

// Structure to represent a path with multiple objectives
struct Path {
    std::vector<int> nodes;
    Distance dist;
    
    Path(int num_objectives) : dist(num_objectives) {}
    Path(const std::vector<int>& n, const Distance& d) : nodes(n), dist(d) {}
};

// Class to represent a Graph
class Graph {
private:
    int V;
    int num_objectives;
    std::vector<std::vector<Edge> > adj;
    
public:
    Graph(int vertices, int objectives) : V(vertices), num_objectives(objectives) {
        adj.resize(vertices);
    }
    
    void addEdge(int src, int dest, const std::vector<int>& weights) {
        if (weights.size() != num_objectives) {
            std::cerr << "Error: Number of weights doesn't match the number of objectives" << std::endl;
            return;
        }
        adj[src].push_back(Edge(src, dest, weights));
    }
    
    const std::vector<std::vector<Edge> >& getAdjList() const {
        return adj;
    }
    
    int getV() const {
        return V;
    }
    
    int getNumObjectives() const {
        return num_objectives;
    }
    
    std::pair<std::vector<int>, std::vector<int> > computeSOSP(int source, int objective_idx) {
        std::vector<int> dist(V, std::numeric_limits<int>::max());
        std::vector<int> pred(V, -1);
        std::vector<bool> visited(V, false);
        std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int> >, std::greater<std::pair<int, int> > > pq;
        
        dist[source] = 0;
        pq.push(std::make_pair(0, source));
        
        while (!pq.empty()) {
            int u = pq.top().second;
            pq.pop();
            
            if (visited[u]) continue;
            visited[u] = true;
            
            for (size_t i = 0; i < adj[u].size(); ++i) {
                const Edge& edge = adj[u][i];
                int v = edge.dest;
                int weight = edge.weights[objective_idx];
                
                if (!visited[v] && dist[u] != std::numeric_limits<int>::max() && 
                    dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    pred[v] = u;
                    pq.push(std::make_pair(dist[v], v));
                }
            }
        }
        
        return std::make_pair(dist, pred);
    }
    
    std::pair<std::vector<int>, std::vector<int> > updateSOSP(int source, int objective_idx, 
                                                            const std::vector<Edge>& new_edges) {
        std::pair<std::vector<int>, std::vector<int> > sosp_result = computeSOSP(source, objective_idx);
        std::vector<int> dist = sosp_result.first;
        std::vector<int> pred = sosp_result.second;
        
        // Add new edges temporarily
        std::vector<std::pair<int, Edge> > added_edges; // Track added edges for removal
        for (size_t i = 0; i < new_edges.size(); ++i) {
            const Edge& edge = new_edges[i];
            adj[edge.src].push_back(edge);
            added_edges.push_back(std::make_pair(edge.src, edge));
        }
        
        // Process new edges
        for (size_t i = 0; i < new_edges.size(); ++i) {
            const Edge& edge = new_edges[i];
            int src = edge.src;
            int dest = edge.dest;
            int weight = edge.weights[objective_idx];
            
            if (dist[src] != std::numeric_limits<int>::max() && 
                dist[src] + weight < dist[dest]) {
                dist[dest] = dist[src] + weight;
                pred[dest] = src;
            }
        }
        
        // Propagate changes
        std::vector<bool> visited(V, false);
        std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int> >, std::greater<std::pair<int, int> > > pq;
        
        for (size_t i = 0; i < new_edges.size(); ++i) {
            const Edge& edge = new_edges[i];
            int dest = edge.dest;
            if (dist[dest] < std::numeric_limits<int>::max()) {
                pq.push(std::make_pair(dist[dest], dest));
            }
        }
        
        while (!pq.empty()) {
            int u = pq.top().second;
            pq.pop();
            
            if (visited[u]) continue;
            visited[u] = true;
            
            for (size_t i = 0; i < adj[u].size(); ++i) {
                const Edge& edge = adj[u][i];
                int v = edge.dest;
                int weight = edge.weights[objective_idx];
                
                if (!visited[v] && dist[u] != std::numeric_limits<int>::max() && 
                    dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    pred[v] = u;
                    pq.push(std::make_pair(dist[v], v));
                }
            }
        }
        
        // Remove temporary edges
        for (size_t i = 0; i < added_edges.size(); ++i) {
            int src = added_edges[i].first;
            const Edge& edge = added_edges[i].second;
            std::vector<Edge>& edges = adj[src];
            for (std::vector<Edge>::iterator it = edges.begin(); it != edges.end(); ++it) {
                if (it->dest == edge.dest && it->weights == edge.weights) {
                    edges.erase(it);
                    break;
                }
            }
        }
        
        return std::make_pair(dist, pred);
    }
    
    std::vector<Path> updateMOSP(int source, const std::vector<Edge>& new_edges) {
        std::vector<std::vector<int> > sosp_distances;
        std::vector<std::vector<int> > sosp_pred;
        
        for (int obj = 0; obj < num_objectives; obj++) {
            std::pair<std::vector<int>, std::vector<int> > sosp_result = updateSOSP(source, obj, new_edges);
            sosp_distances.push_back(sosp_result.first);
            sosp_pred.push_back(sosp_result.second);
        }
        
        Graph combined_graph(V, 1);
        
        for (int u = 0; u < V; u++) {
            for (size_t i = 0; i < adj[u].size(); ++i) {
                const Edge& edge = adj[u][i];
                int v = edge.dest;
                int count = 0;
                
                for (int obj = 0; obj < num_objectives; obj++) {
                    if (sosp_distances[obj][u] != std::numeric_limits<int>::max() && 
                        sosp_distances[obj][v] != std::numeric_limits<int>::max() && 
                        sosp_distances[obj][u] + edge.weights[obj] == sosp_distances[obj][v]) {
                        count++;
                    }
                }
                
                int adjusted_weight = num_objectives - count + 1;
                combined_graph.addEdge(u, v, std::vector<int>(1, adjusted_weight));
            }
        }
        
        for (size_t i = 0; i < new_edges.size(); ++i) {
            const Edge& edge = new_edges[i];
            int u = edge.src;
            int v = edge.dest;
            int count = 0;
            
            for (int obj = 0; obj < num_objectives; obj++) {
                if (sosp_distances[obj][u] != std::numeric_limits<int>::max() && 
                    sosp_distances[obj][v] != std::numeric_limits<int>::max() && 
                    sosp_distances[obj][u] + edge.weights[obj] == sosp_distances[obj][v]) {
                    count++;
                }
            }
            
            int adjusted_weight = num_objectives - count + 1;
            combined_graph.addEdge(u, v, std::vector<int>(1, adjusted_weight));
        }
        
        std::pair<std::vector<int>, std::vector<int> > combined_result = combined_graph.computeSOSP(source, 0);
        std::vector<int> combined_dist = combined_result.first;
        std::vector<int> combined_pred = combined_result.second;
        
        std::vector<Path> pareto_paths;
        
        for (int v = 0; v < V; v++) {
            if (combined_dist[v] != std::numeric_limits<int>::max()) {
                std::vector<int> nodes;
                int current = v;
                while (current != -1) {
                    nodes.push_back(current);
                    current = sosp_pred[0][current];
                }
                std::reverse(nodes.begin(), nodes.end());
                
                Path path(num_objectives);
                path.nodes = nodes;
                
                for (int obj = 0; obj < num_objectives; obj++) {
                    path.dist.values[obj] = sosp_distances[obj][v];
                }
                
                pareto_paths.push_back(path);
            }
        }
        
        return pareto_paths;
    }
};

// Load graph with corrected vertex mapping
Graph loadFacebookGraph(const std::string& filename, int num_objectives) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }
    
    std::string line;
    std::unordered_set<int> vertices;
    std::vector<std::tuple<int, int, int> > edge_pairs;
    
    while (std::getline(file, line)) {
        if (line[0] == '#') continue;
        
        std::istringstream iss(line);
        int src, dest, weight;
        if (!(iss >> src >> dest >> weight)) continue;
        
        vertices.insert(src);
        vertices.insert(dest);
        edge_pairs.push_back(std::make_tuple(src, dest, weight));
    }
    
    file.close();
    
    // Create graph without remapping (use original vertex IDs)
    int V = vertices.empty() ? 0 : *std::max_element(vertices.begin(), vertices.end()) + 1;
    Graph graph(V, num_objectives);
    
    // Add edges with original IDs
    for (size_t i = 0; i < edge_pairs.size(); ++i) {
        int src = std::get<0>(edge_pairs[i]);
        int dest = std::get<1>(edge_pairs[i]);
        int weight = std::get<2>(edge_pairs[i]);
        
        std::vector<int> weights(num_objectives, weight);
        graph.addEdge(src, dest, weights);
        graph.addEdge(dest, src, weights);
    }
    
    return graph;
}

// Generate specific new edges
std::vector<Edge> generateNewEdges(const Graph& graph, int num_new_edges) {
    int num_objectives = graph.getNumObjectives();
    std::vector<Edge> new_edges;
    new_edges.push_back(Edge(1, 4, std::vector<int>(num_objectives, 3))); // Edge 1→4, weight 3
    new_edges.push_back(Edge(0, 3, std::vector<int>(num_objectives, 8))); // Edge 0→3, weight 8
    return new_edges;
}

// Save results with full paths
void saveResults(const std::string& filename, 
                const std::vector<Path>& paths, 
                double execution_time) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening output file: " << filename << std::endl;
        return;
    }
    
    file << "Execution Time: " << execution_time << " seconds" << std::endl;
    file << "Number of Pareto-optimal paths: " << paths.size() << std::endl << std::endl;
    
    for (size_t p = 0; p < paths.size(); ++p) {
        const Path& path = paths[p];
        file << "Path from " << path.nodes.front() << " to " << path.nodes.back() << ": ";
        file << "Nodes: [";
        for (size_t i = 0; i < path.nodes.size(); i++) {
            file << path.nodes[i];
            if (i < path.nodes.size() - 1) {
                file << ", ";
            }
        }
        file << "] ";
        
        file << "Objectives: [";
        for (size_t i = 0; i < path.dist.values.size(); i++) {
            file << path.dist.values[i];
            if (i < path.dist.values.size() - 1) {
                file << ", ";
            }
        }
        file << "]" << std::endl;
    }
    
    file.close();
}

int main(int argc, char* argv[]) {
    std::string graph_file = "sample_graph.txt";
    int num_objectives = 5;
    int source_vertex = 0;
    int num_new_edges = 2;
    std::string output_file = "results.txt";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--graph" && i + 1 < argc) {
            graph_file = argv[++i];
        } else if (arg == "--objectives" && i + 1 < argc) {
            num_objectives = std::stoi(argv[++i]);
        } else if (arg == "--source" && i + 1 < argc) {
            source_vertex = std::stoi(argv[++i]);
        } else if (arg == "--new-edges" && i + 1 < argc) {
            num_new_edges = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        }
    }
    
    std::cout << "Starting MOSP algorithm with:" << std::endl;
    std::cout << "- Graph file: " << graph_file << std::endl;
    std::cout << "- Number of objectives: " << num_objectives << std::endl;
    std::cout << "- Source vertex: " << source_vertex << std::endl;
    std::cout << "- Number of new edges: " << num_new_edges << std::endl;
    
    Graph graph = loadFacebookGraph(graph_file, num_objectives);
    std::cout << "Loaded graph with " << graph.getV() << " vertices" << std::endl;
    
    std::vector<Edge> new_edges = generateNewEdges(graph, num_new_edges);
    std::cout << "Generated " << new_edges.size() << " new edges" << std::endl;
    
    std::cout << "Starting MOSP update computation..." << std::endl;
    
    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<Path> paths = graph.updateMOSP(source_vertex, new_edges);
    
    std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "MOSP update completed in " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Found " << paths.size() << " Pareto-optimal paths" << std::endl;
    
    saveResults(output_file, paths, elapsed.count());
    
    int count = 0;
    for (size_t p = 0; p < paths.size(); ++p) {
        const Path& path = paths[p];
        if (count >= 5) break;
        
        std::cout << "Path to node " << path.nodes.back() << ": ";
        std::cout << "Nodes: [";
        for (size_t i = 0; i < path.nodes.size(); i++) {
            std::cout << path.nodes[i];
            if (i < path.nodes.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "] ";
        for (int obj = 0; obj < num_objectives; obj++) {
            std::cout << "Obj " << obj << "=" << path.dist.values[obj] << " ";
        }
        std::cout << std::endl;
        
        count++;
    }
    
    return 0;
}
