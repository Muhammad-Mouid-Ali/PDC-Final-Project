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
#include <mpi.h>
#include <metis.h>

// Custom hash for pair<int, int>
struct pair_hash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

// Enum for edge operation type
enum EdgeOperation {
    INSERT,
    DELETE
};

// Structure to represent an edge
struct Edge {
    int src, dest;
    std::vector<int> weights;

    Edge() : src(0), dest(0), weights() {}
    Edge(int s, int d, const std::vector<int>& w) : src(s), dest(d), weights(w) {}
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
            if (values[i] > other.values[i]) return false;
            if (values[i] < other.values[i]) at_least_one_better = true;
        }
        return at_least_one_better;
    }
    
    bool operator==(const Distance& other) const {
        if (values.size() != other.values.size()) return false;
        for (size_t i = 0; i < values.size(); i++) {
            if (values[i] != other.values[i]) return false;
        }
        return true;
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
    std::vector<std::vector<Edge>> adj;
    
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
    
    void removeEdge(int src, int dest) {
        auto& edges = adj[src];
        edges.erase(std::remove_if(edges.begin(), edges.end(),
                                   [dest](const Edge& e) { return e.dest == dest; }),
                    edges.end());
    }
    
    const std::vector<std::vector<Edge>>& getAdjList() const {
        return adj;
    }
    
    int getV() const {
        return V;
    }
    
    int getNumObjectives() const {
        return num_objectives;
    }
    
    std::pair<std::vector<int>, std::vector<int>> computeSOSP(int source, int objective_idx) {
        std::vector<int> dist(V, std::numeric_limits<int>::max());
        std::vector<int> pred(V, -1);
        std::vector<bool> visited(V, false);
        std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;
        
        dist[source] = 0;
        pq.push({0, source});
        
        while (!pq.empty()) {
            int u = pq.top().second;
            int u_dist = pq.top().first;
            pq.pop();
            
            if (visited[u]) continue;
            visited[u] = true;
            
            for (size_t i = 0; i < adj[u].size(); i++) {
                const Edge& edge = adj[u][i];
                int v = edge.dest;
                int weight = edge.weights[objective_idx];
                int new_dist = (u_dist == std::numeric_limits<int>::max() || weight == std::numeric_limits<int>::max()) 
                              ? std::numeric_limits<int>::max() : u_dist + weight;
                
                if (!visited[v] && new_dist < dist[v]) {
                    dist[v] = new_dist;
                    pred[v] = u;
                    pq.push({new_dist, v});
                }
            }
        }
        
        return {dist, pred};
    }
    
    std::pair<std::vector<int>, std::vector<int>> computeBellmanFord(int source, int objective_idx) {
        std::vector<int> dist(V, std::numeric_limits<int>::max());
        std::vector<int> pred(V, -1);
        
        dist[source] = 0;
        
        for (int i = 0; i < V - 1; i++) {
            bool updated = false;
            for (int u = 0; u < V; u++) {
                for (const Edge& edge : adj[u]) {
                    int v = edge.dest;
                    int weight = edge.weights[objective_idx];
                    int new_dist = (dist[u] == std::numeric_limits<int>::max() || weight == std::numeric_limits<int>::max()) 
                                  ? std::numeric_limits<int>::max() : dist[u] + weight;
                    if (new_dist < dist[v]) {
                        dist[v] = new_dist;
                        pred[v] = u;
                        updated = true;
                    }
                }
            }
            if (!updated) break;
        }
        
        return {dist, pred};
    }
    
    std::pair<std::vector<int>, std::vector<int>> updateLocalSOSP(int source, int objective_idx, 
                                                                const std::vector<Edge>& update_edges,
                                                                EdgeOperation op,
                                                                const std::vector<int>& partition,
                                                                int part_id, const std::vector<int>& boundary_vertices) {
        std::vector<int> dist(V, std::numeric_limits<int>::max());
        std::vector<int> pred(V, -1);
        std::vector<bool> visited(V, false);
        std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<>> pq;
        
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        
        if (partition[source] == part_id) {
            dist[source] = 0;
            pq.push({0, source});
            if (rank == part_id) {
                std::cout << "Rank " << rank << ": Source vertex " << source << " initialized with distance 0 for objective " << objective_idx << std::endl;
            }
        }
        
        for (int v : boundary_vertices) {
            if (partition[v] == part_id) {
                dist[v] = std::numeric_limits<int>::max();
                pred[v] = -1;
                pq.push({std::numeric_limits<int>::max(), v});
            }
        }
        
        int visited_count = 0;
        while (!pq.empty()) {
            int u = pq.top().second;
            int u_dist = pq.top().first;
            pq.pop();
            
            if (visited[u] || partition[u] != part_id) continue;
            visited[u] = true;
            visited_count++;
            
            for (const Edge& edge : adj[u]) {
                int v = edge.dest;
                if (op == DELETE) {
                    bool is_deleted = false;
                    for (const Edge& del_edge : update_edges) {
                        if (del_edge.src == edge.src && del_edge.dest == edge.dest) {
                            is_deleted = true;
                            break;
                        }
                    }
                    if (is_deleted) continue;
                }
                
                int weight = edge.weights[objective_idx];
                int new_dist = (u_dist == std::numeric_limits<int>::max() || weight == std::numeric_limits<int>::max()) 
                              ? std::numeric_limits<int>::max() : u_dist + weight;
                
                if (!visited[v] && new_dist < dist[v]) {
                    dist[v] = new_dist;
                    pred[v] = u;
                    pq.push({new_dist, v});
                }
            }
            
            if (op == INSERT) {
                for (const Edge& edge : update_edges) {
                    if (edge.src == u && (partition[edge.dest] == part_id || std::find(boundary_vertices.begin(), boundary_vertices.end(), edge.dest) != boundary_vertices.end())) {
                        int v = edge.dest;
                        int weight = edge.weights[objective_idx];
                        int new_dist = (u_dist == std::numeric_limits<int>::max() || weight == std::numeric_limits<int>::max()) 
                                      ? std::numeric_limits<int>::max() : u_dist + weight;
                        if (!visited[v] && new_dist < dist[v]) {
                            dist[v] = new_dist;
                            pred[v] = u;
                            pq.push({new_dist, v});
                        }
                    }
                }
            }
        }
        
        if (rank == part_id) {
            std::cout << "Rank " << rank << ": Visited " << visited_count << " vertices for objective " << objective_idx << std::endl;
        }
        
        return {dist, pred};
    }
    
    std::vector<Path> updateMOSP(int source, const std::vector<Edge>& update_edges, EdgeOperation op) {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // Partition the graph
        auto [partition, boundary_vertices] = partitionGraph(size);
        
        if (rank == 0) {
            std::cout << "Partitioned graph into " << size << " partitions with " 
                      << boundary_vertices.size() << " boundary vertices" << std::endl;
        }
        
        // Apply edge updates to local adjacency list
        if (op == INSERT) {
            for (const Edge& edge : update_edges) {
                addEdge(edge.src, edge.dest, edge.weights);
                addEdge(edge.dest, edge.src, edge.weights);
            }
        } else if (op == DELETE) {
            for (const Edge& edge : update_edges) {
                removeEdge(edge.src, edge.dest);
                removeEdge(edge.dest, edge.src);
            }
        }
        
        std::vector<std::vector<int>> local_distances(num_objectives, std::vector<int>(V, std::numeric_limits<int>::max()));
        std::vector<std::vector<int>> local_pred(num_objectives, std::vector<int>(V, -1));
        
        if (partition[source] == rank) {
            for (int obj = 0; obj < num_objectives; obj++) {
                local_distances[obj][source] = 0;
            }
        }
        
        for (int obj = 0; obj < num_objectives; obj++) {
            auto sosp_result = updateLocalSOSP(source, obj, update_edges, op, partition, rank, boundary_vertices);
            local_distances[obj] = sosp_result.first;
            local_pred[obj] = sosp_result.second;
        }
        
        // Gather all local distances
        std::vector<int> send_buffer;
        for (int v = 0; v < V; v++) {
            for (int obj = 0; obj < num_objectives; obj++) {
                send_buffer.push_back(local_distances[obj][v]);
                send_buffer.push_back(local_pred[obj][v]);
            }
        }
        
        std::vector<int> recv_counts(size), displs(size);
        int send_count = send_buffer.size();
        MPI_Allgather(&send_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        int total_recv = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
        std::vector<int> recv_buffer(total_recv);
        
        displs[0] = 0;
        for (int p = 1; p < size; p++) {
            displs[p] = displs[p-1] + recv_counts[p-1];
        }
        
        MPI_Allgatherv(send_buffer.data(), send_count, MPI_INT,
                       recv_buffer.data(), recv_counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);
        
        if (rank == 0) {
            std::cout << "Rank 0: Received boundary data, total size = " << total_recv << std::endl;
        }
        
        std::vector<std::vector<int>> sosp_distances(num_objectives, std::vector<int>(V, std::numeric_limits<int>::max()));
        std::vector<std::vector<int>> sosp_pred(num_objectives, std::vector<int>(V, -1));
        
        for (int obj = 0; obj < num_objectives; obj++) {
            sosp_distances[obj][source] = 0;
        }
        
        for (int p = 0; p < size; p++) {
            int offset = displs[p];
            for (int v = 0; v < V; v++) {
                for (int obj = 0; obj < num_objectives; obj++) {
                    int dist_idx = offset + v * 2 * num_objectives + obj * 2;
                    int pred_idx = dist_idx + 1;
                    if (dist_idx < total_recv && recv_buffer[dist_idx] < sosp_distances[obj][v]) {
                        sosp_distances[obj][v] = recv_buffer[dist_idx];
                        sosp_pred[obj][v] = recv_buffer[pred_idx];
                    }
                }
            }
        }
        
        std::vector<Path> pareto_paths;
        if (rank == 0) {
            std::cout << "All distances from source:" << std::endl;
            for (int v = 0; v < V; v++) {
                std::cout << "Vertex " << v << ": [";
                for (int obj = 0; obj < num_objectives; obj++) {
                    if (sosp_distances[obj][v] == std::numeric_limits<int>::max()) {
                        std::cout << "INF";
                    } else {
                        std::cout << sosp_distances[obj][v];
                    }
                    if (obj < num_objectives - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
            
            // Construct adjacency list for path reconstruction
            std::vector<std::vector<std::pair<int, std::vector<int>>>> valid_edges(V);
            int edge_count = 0;
            for (int u = 0; u < V; u++) {
                for (const Edge& edge : adj[u]) {
                    int v = edge.dest;
                    bool valid = true;
                    for (int obj = 0; obj < num_objectives; obj++) {
                        if (sosp_distances[obj][u] == std::numeric_limits<int>::max() ||
                            sosp_distances[obj][v] == std::numeric_limits<int>::max() ||
                            edge.weights[obj] == std::numeric_limits<int>::max() ||
                            sosp_distances[obj][u] + edge.weights[obj] > sosp_distances[obj][v]) {
                            valid = false;
                            break;
                        }
                    }
                    if (valid) {
                        valid_edges[u].push_back({v, edge.weights});
                        edge_count++;
                    }
                }
            }
            std::cout << "Valid edges for path reconstruction: " << edge_count << std::endl;
            
            // Reconstruct paths using DFS
            std::vector<Path> candidate_paths;
            std::vector<int> current_path = {source};
            Distance current_dist(num_objectives, 0);
            std::vector<bool> visited(V, false);
            visited[source] = true;
            
            std::function<void(int)> dfs = [&](int u) {
                for (const auto& [v, weights] : valid_edges[u]) {
                    if (!visited[v]) {
                        Distance new_dist(num_objectives);
                        bool valid_path = true;
                        for (int obj = 0; obj < num_objectives; obj++) {
                            if (current_dist.values[obj] == std::numeric_limits<int>::max() ||
                                weights[obj] == std::numeric_limits<int>::max()) {
                                new_dist.values[obj] = std::numeric_limits<int>::max();
                            } else {
                                new_dist.values[obj] = current_dist.values[obj] + weights[obj];
                                if (new_dist.values[obj] > sosp_distances[obj][v]) {
                                    valid_path = false;
                                }
                            }
                        }
                        if (!valid_path) continue;
                        
                        current_path.push_back(v);
                        visited[v] = true;
                        Distance prev_dist = current_dist;
                        current_dist = new_dist;
                        
                        if (sosp_distances[0][v] != std::numeric_limits<int>::max()) {
                            Path path(num_objectives);
                            path.nodes = current_path;
                            path.dist = Distance(num_objectives);
                            for (int obj = 0; obj < num_objectives; obj++) {
                                path.dist.values[obj] = sosp_distances[obj][v];
                            }
                            candidate_paths.push_back(path);
                        }
                        
                        dfs(v);
                        
                        current_path.pop_back();
                        visited[v] = false;
                        current_dist = prev_dist;
                    }
                }
            };
            
            dfs(source);
            
            std::cout << "Candidate paths before filtering (" << candidate_paths.size() << "):" << std::endl;
            for (size_t i = 0; i < candidate_paths.size(); i++) {
                const Path& path = candidate_paths[i];
                std::cout << "Path " << i << ": ";
                for (int node : path.nodes) std::cout << node << " ";
                std::cout << "Objectives: ";
                for (int obj : path.dist.values) {
                    if (obj == std::numeric_limits<int>::max()) {
                        std::cout << "INF ";
                    } else {
                        std::cout << obj << " ";
                    }
                }
                std::cout << std::endl;
            }
            
            // Filter Pareto-optimal paths
            for (const Path& path : candidate_paths) {
                bool dominated = false;
                for (auto it = pareto_paths.begin(); it != pareto_paths.end();) {
                    if (it->dist.dominates(path.dist)) {
                        std::cout << "Path to " << path.nodes.back() << " dominated by path to " << it->nodes.back() << std::endl;
                        dominated = true;
                        break;
                    } else if (path.dist.dominates(it->dist)) {
                        std::cout << "Path to " << path.nodes.back() << " dominates path to " << it->nodes.back() << std::endl;
                        it = pareto_paths.erase(it);
                    } else {
                        ++it;
                    }
                }
                if (!dominated) {
                    std::cout << "Adding non-dominated path to " << path.nodes.back() << std::endl;
                    pareto_paths.push_back(path);
                }
            }
            
            std::cout << "Final Pareto-optimal paths (" << pareto_paths.size() << "):" << std::endl;
            for (size_t i = 0; i < pareto_paths.size(); i++) {
                const Path& path = pareto_paths[i];
                std::cout << "Path " << i << ": ";
                for (int node : path.nodes) std::cout << node << " ";
                std::cout << "Objectives: ";
                for (int obj : path.dist.values) {
                    if (obj == std::numeric_limits<int>::max()) {
                        std::cout << "INF ";
                    } else {
                        std::cout << obj << " ";
                    }
                }
                std::cout << std::endl;
            }
        }
        
        int num_paths = pareto_paths.size();
        MPI_Bcast(&num_paths, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        return pareto_paths;
    }
    
    std::pair<std::vector<int>, std::vector<int>> computeMOSPWithBellmanFord(int source, int objective_idx) {
        std::vector<int> dist(V, std::numeric_limits<int>::max());
        std::vector<int> pred(V, -1);
        
        dist[source] = 0;
        
        for (int i = 0; i < V - 1; i++) {
            bool any_change = false;
            for (int u = 0; u < V; u++) {
                if (dist[u] == std::numeric_limits<int>::max()) continue;
                
                for (const Edge& edge : adj[u]) {
                    int v = edge.dest;
                    int weight = edge.weights[objective_idx];
                    
                    if (dist[u] + weight < dist[v]) {
                        dist[v] = dist[u] + weight;
                        pred[v] = u;
                        any_change = true;
                    }
                }
            }
            
            if (!any_change) break;
        }
        
        return {dist, pred};
    }
    
    std::pair<std::vector<int>, std::vector<int>> partitionGraph(int num_partitions) {
        std::vector<int> partition(V);
        std::vector<int> boundary_vertices;
        
        idx_t nvtxs = V;
        idx_t ncon = 1;
        idx_t nparts = num_partitions;
        idx_t objval;
        std::vector<idx_t> xadj(V + 1, 0);
        std::vector<idx_t> adjncy;
        std::vector<idx_t> part(V);
        
        for (int u = 0; u < V; u++) {
            xadj[u] = adjncy.size();
            for (const Edge& edge : adj[u]) {
                adjncy.push_back(edge.dest);
            }
        }
        xadj[V] = adjncy.size();
        
        int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), NULL, NULL, NULL,
                                      &nparts, NULL, NULL, NULL, &objval, part.data());
        if (ret != METIS_OK) {
            std::cerr << "METIS partitioning failed" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        for (int v = 0; v < V; v++) {
            partition[v] = part[v];
        }
        
        std::unordered_set<int> boundary_set;
        for (int u = 0; u < V; u++) {
            for (const Edge& edge : adj[u]) {
                if (partition[u] != partition[edge.dest]) {
                    boundary_set.insert(u);
                    boundary_set.insert(edge.dest);
                }
            }
        }
        boundary_vertices.assign(boundary_set.begin(), boundary_set.end());
        
        return {partition, boundary_vertices};
    }
};

// Load graph with MPI broadcasting
Graph loadFacebookGraphMPI(const std::string& filename, int num_objectives, int rank, int size) {
    Graph graph(0, num_objectives);
    int V = 0;
    std::vector<std::tuple<int, int, int>> edge_pairs;
    
    if (rank == 0) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::string line;
        std::unordered_set<int> vertices;
        while (std::getline(file, line)) {
            if (line[0] == '#') continue;
            std::istringstream iss(line);
            int src, dest, weight;
            if (!(iss >> src >> dest >> weight)) continue;
            vertices.insert(src);
            vertices.insert(dest);
            edge_pairs.push_back({src, dest, weight});
        }
        file.close();
        V = vertices.empty() ? 0 : *std::max_element(vertices.begin(), vertices.end()) + 1;
    }
    
    MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);
    graph = Graph(V, num_objectives);
    
    int edge_count = edge_pairs.size();
    MPI_Bcast(&edge_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        edge_pairs.resize(edge_count);
    }
    
    std::vector<int> edge_data(edge_count * 3);
    if (rank == 0) {
        for (size_t i = 0; i < edge_pairs.size(); i++) {
            edge_data[i * 3] = std::get<0>(edge_pairs[i]);
            edge_data[i * 3 + 1] = std::get<1>(edge_pairs[i]);
            edge_data[i * 3 + 2] = std::get<2>(edge_pairs[i]);
        }
    }
    MPI_Bcast(edge_data.data(), edge_count * 3, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        for (int i = 0; i < edge_count; i++) {
            edge_pairs[i] = {edge_data[i * 3], edge_data[i * 3 + 1], edge_data[i * 3 + 2]};
        }
    }
    
    for (const auto& [src, dest, weight] : edge_pairs) {
        std::vector<int> weights(num_objectives);
        for (int i = 0; i < num_objectives; i++) {
            weights[i] = weight + i; // Vary weights slightly per objective
        }
        graph.addEdge(src, dest, weights);
        graph.addEdge(dest, src, weights);
    }
    
    return graph;
}

// Generate new edges and random edge deletions
std::pair<std::vector<Edge>, std::vector<Edge>> generateEdgeUpdates(const Graph& graph, int num_new_edges, int num_delete_edges) {
    int num_objectives = graph.getNumObjectives();
    int V = graph.getV();
    std::vector<Edge> new_edges;
    std::vector<Edge> deleted_edges;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> vertex_dist(0, V - 1);
    std::uniform_int_distribution<> weight_dist(1, 10);
    
    std::unordered_set<std::pair<int, int>, pair_hash> existing_edges;
    for (int u = 0; u < V; u++) {
        for (const Edge& edge : graph.getAdjList()[u]) {
            if (u < edge.dest) {
                existing_edges.emplace(u, edge.dest);
            }
        }
    }
    
    // Generate new edges
    for (int i = 0; i < num_new_edges; i++) {
        int src, dest;
        do {
            src = vertex_dist(gen);
            dest = vertex_dist(gen);
        } while (src == dest || existing_edges.count({std::min(src, dest), std::max(src, dest)}));
        
        std::vector<int> weights(num_objectives);
        for (int j = 0; j < num_objectives; j++) {
            weights[j] = weight_dist(gen) + j; // Different weights per objective
        }
        new_edges.emplace_back(src, dest, weights);
        existing_edges.emplace(std::min(src, dest), std::max(src, dest));
    }
    
    // Generate edges for deletion, ensuring connectivity
    std::vector<std::pair<int, int>> all_edges;
    for (int u = 0; u < V; u++) {
        for (const Edge& edge : graph.getAdjList()[u]) {
            if (u < edge.dest) {
                all_edges.emplace_back(u, edge.dest);
            }
        }
    }
    
    std::shuffle(all_edges.begin(), all_edges.end(), gen);
    num_delete_edges = std::min(num_delete_edges, (int)all_edges.size());
    
    // Simple connectivity check using DFS
    auto is_connected = [&](const std::vector<std::pair<int, int>>& edges_to_remove) {
        std::vector<std::vector<int>> temp_adj(V);
        std::unordered_set<std::pair<int, int>, pair_hash> remove_set;
        for (const auto& e : edges_to_remove) {
            remove_set.emplace(std::min(e.first, e.second), std::max(e.first, e.second));
        }
        for (int u = 0; u < V; u++) {
            for (const Edge& edge : graph.getAdjList()[u]) {
                if (!remove_set.count({std::min(u, edge.dest), std::max(u, edge.dest)})) {
                    temp_adj[u].push_back(edge.dest);
                }
            }
        }
        std::vector<bool> visited(V, false);
        std::function<void(int)> dfs = [&](int v) {
            visited[v] = true;
            for (int u : temp_adj[v]) {
                if (!visited[u]) dfs(u);
            }
        };
        dfs(0);
        return std::all_of(visited.begin(), visited.end(), [](bool v) { return v; });
    };
    
    std::vector<std::pair<int, int>> selected_edges;
    for (int i = 0; i < num_delete_edges && i < (int)all_edges.size(); i++) {
        selected_edges.push_back(all_edges[i]);
        if (!is_connected(selected_edges)) {
            selected_edges.pop_back();
        }
    }
    
    for (const auto& [src, dest] : selected_edges) {
        std::vector<int> weights(num_objectives, 0);
        for (const Edge& edge : graph.getAdjList()[src]) {
            if (edge.dest == dest) {
                weights = edge.weights;
                break;
            }
        }
        deleted_edges.emplace_back(src, dest, weights);
    }
    
    return {new_edges, deleted_edges};
}

// Save results
void saveResults(const std::string& filename, 
                const std::vector<Path>& insert_paths, 
                const std::vector<std::vector<int>>& insert_distances,
                const std::vector<Path>& delete_paths, 
                const std::vector<std::vector<int>>& delete_distances,
                int num_objectives,
                double insert_time,
                double delete_time) {
    std::ofstream file(filename, std::ios::app); // Append to avoid overwriting
    if (!file.is_open()) {
        std::cerr << "Error opening output file: " << filename << std::endl;
        return;
    }
    
    file << "=== Edge Insertion ===\n";
    file << "Execution Time: " << insert_time << " seconds\n";
    file << "Number of Pareto-optimal paths: " << insert_paths.size() << "\n\n";
    
    file << "Pareto-optimal Paths:\n";
    for (size_t p = 0; p < insert_paths.size(); ++p) {
        const Path& path = insert_paths[p];
        file << "Path from " << path.nodes.front() << " to " << path.nodes.back() << ": ";
        file << "Nodes: [";
        for (size_t i = 0; i < path.nodes.size(); i++) {
            file << path.nodes[i];
            if (i < path.nodes.size() - 1) file << ", ";
        }
        file << "] Objectives: [";
        for (size_t i = 0; i < path.dist.values.size(); i++) {
            if (path.dist.values[i] == std::numeric_limits<int>::max()) {
                file << "INF";
            } else {
                file << path.dist.values[i];
            }
            if (i < path.dist.values.size() - 1) file << ", ";
        }
        file << "]\n";
    }
    
    file << "\nDistances from Source (Insertion):\n";
    for (size_t v = 0; v < insert_distances[0].size(); v++) {
        file << "Vertex " << v << ": [";
        for (int obj = 0; obj < num_objectives; obj++) {
            if (insert_distances[obj][v] == std::numeric_limits<int>::max()) {
                file << "INF";
            } else {
                file << insert_distances[obj][v];
            }
            if (obj < num_objectives - 1) file << ", ";
        }
        file << "]\n";
    }
    
    file << "\n=== Edge Deletion ===\n";
    file << "Execution Time: " << delete_time << " seconds\n";
    file << "Number of Pareto-optimal paths: " << delete_paths.size() << "\n\n";
    
    file << "Pareto-optimal Paths:\n";
    for (size_t p = 0; p < delete_paths.size(); ++p) {
        const Path& path = delete_paths[p];
        file << "Path from " << path.nodes.front() << " to " << path.nodes.back() << ": ";
        file << "Nodes: [";
        for (size_t i = 0; i < path.nodes.size(); i++) {
            file << path.nodes[i];
            if (i < path.nodes.size() - 1) file << ", ";
        }
        file << "] Objectives: [";
        for (size_t i = 0; i < path.dist.values.size(); i++) {
            if (path.dist.values[i] == std::numeric_limits<int>::max()) {
                file << "INF";
            } else {
                file << path.dist.values[i];
            }
            if (i < path.dist.values.size() - 1) file << ", ";
        }
        file << "]\n";
    }
    
    file << "\nDistances from Source (Deletion):\n";
    for (size_t v = 0; v < delete_distances[0].size(); v++) {
        file << "Vertex " << v << ": [";
        for (int obj = 0; obj < num_objectives; obj++) {
            if (delete_distances[obj][v] == std::numeric_limits<int>::max()) {
                file << "INF";
            } else {
                file << delete_distances[obj][v];
            }
            if (obj < num_objectives - 1) file << ", ";
        }
        file << "]\n";
    }
    
    file << "\n";
    file.close();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    std::string graph_file = "weightedfacebook_graph.txt";
    int num_objectives = 5;
    int source_vertex = 0;
    int num_new_edges = 10;
    int num_delete_edges = 2;
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
        } else if (arg == "--delete-edges" && i + 1 < argc) {
            num_delete_edges = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        }
    }
    
    if (rank == 0) {
        std::ofstream file(output_file, std::ios::trunc); // Clear output file
        file.close();
        std::cout << "Starting Parallel MOSP algorithm with:" << std::endl;
        std::cout << "- MPI Processes: " << size << std::endl;
        std::cout << "- Graph file: " << graph_file << std::endl;
        std::cout << "- Number of objectives: " << num_objectives << std::endl;
        std::cout << "- Source vertex: " << source_vertex << std::endl;
        std::cout << "- Number of new edges: " << num_new_edges << std::endl;
        std::cout << "- Number of edges to delete: " << num_delete_edges << std::endl;
    }
    
    Graph graph = loadFacebookGraphMPI(graph_file, num_objectives, rank, size);
    
    if (rank == 0) {
        std::cout << "Loaded graph with " << graph.getV() << " vertices" << std::endl;
    }
    
    auto [new_edges, deleted_edges] = generateEdgeUpdates(graph, num_new_edges, num_delete_edges);
    
    if (rank == 0) {
        std::cout << "Generated " << new_edges.size() << " new edges and " << deleted_edges.size() << " edges for deletion" << std::endl;
        std::cout << "Sample new edge weights:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), new_edges.size()); i++) {
            std::cout << "Edge " << new_edges[i].src << "->" << new_edges[i].dest << ": ";
            for (int w : new_edges[i].weights) std::cout << w << " ";
            std::cout << std::endl;
        }
    }
    
    // Edge Insertion
    if (rank == 0) {
        std::cout << "Starting parallel MOSP update for edge insertion..." << std::endl;
    }
    
    std::chrono::high_resolution_clock::time_point insert_start_time;
    if (rank == 0) {
        insert_start_time = std::chrono::high_resolution_clock::now();
    }
    
    std::vector<Path> insert_paths = graph.updateMOSP(source_vertex, new_edges, EdgeOperation::INSERT);
    
    double insert_execution_time = 0.0;
    std::vector<std::vector<int>> insert_distances(num_objectives, std::vector<int>(graph.getV(), std::numeric_limits<int>::max()));
    if (rank == 0) {
        std::chrono::high_resolution_clock::time_point insert_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> insert_elapsed = insert_end_time - insert_start_time;
        insert_execution_time = insert_elapsed.count();
        
        std::cout << "MOSP update for insertion completed in " << insert_execution_time << " seconds" << std::endl;
        std::cout << "Found " << insert_paths.size() << " Pareto-optimal paths" << std::endl;
        
        for (int obj = 0; obj < num_objectives; obj++) {
            auto [dist, pred] = graph.computeSOSP(source_vertex, obj);
            insert_distances[obj] = dist;
        }
        
        for (size_t p = 0; p < insert_paths.size(); ++p) {
            const Path& path = insert_paths[p];
            std::cout << "Path from " << path.nodes.front() << " to " << path.nodes.back() << ": ";
            std::cout << "Nodes: [";
            for (size_t i = 0; i < path.nodes.size(); i++) {
                std::cout << path.nodes[i];
                if (i < path.nodes.size() - 1) std::cout << ", ";
            }
            std::cout << "] Objectives: [";
            for (size_t i = 0; i < path.dist.values.size(); i++) {
                if (path.dist.values[i] == std::numeric_limits<int>::max()) {
                    std::cout << "INF";
                } else {
                    std::cout << path.dist.values[i];
                }
                if (i < path.dist.values.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
    }
    
    // Edge Deletion (reload graph to reset state)
    graph = loadFacebookGraphMPI(graph_file, num_objectives, rank, size);
    
    if (rank == 0) {
        std::cout << "\nStarting parallel MOSP update for edge deletion..." << std::endl;
    }
    
    std::chrono::high_resolution_clock::time_point delete_start_time;
    if (rank == 0) {
        delete_start_time = std::chrono::high_resolution_clock::now();
    }
    
    std::vector<Path> delete_paths = graph.updateMOSP(source_vertex, deleted_edges, EdgeOperation::DELETE);
    
    double delete_execution_time = 0.0;
    std::vector<std::vector<int>> delete_distances(num_objectives, std::vector<int>(graph.getV(), std::numeric_limits<int>::max()));
    if (rank == 0) {
        std::chrono::high_resolution_clock::time_point delete_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> delete_elapsed = delete_end_time - delete_start_time;
        delete_execution_time = delete_elapsed.count();
        
        std::cout << "MOSP update for deletion completed in " << delete_execution_time << " seconds" << std::endl;
        std::cout << "Found " << delete_paths.size() << " Pareto-optimal paths" << std::endl;
        
        for (int obj = 0; obj < num_objectives; obj++) {
            auto [dist, pred] = graph.computeSOSP(source_vertex, obj);
            delete_distances[obj] = dist;
        }
        
        for (size_t p = 0; p < delete_paths.size(); ++p) {
            const Path& path = delete_paths[p];
            std::cout << "Path from " << path.nodes.front() << " to " << path.nodes.back() << ": ";
            std::cout << "Nodes: [";
            for (size_t i = 0; i < path.nodes.size(); i++) {
                std::cout << path.nodes[i];
                if (i < path.nodes.size() - 1) std::cout << ", ";
            }
            std::cout << "] Objectives: [";
            for (size_t i = 0; i < path.dist.values.size(); i++) {
                if (path.dist.values[i] == std::numeric_limits<int>::max()) {
                    std::cout << "INF";
                } else {
                    std::cout << path.dist.values[i];
                }
                if (i < path.dist.values.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
        
        saveResults(output_file, insert_paths, insert_distances, delete_paths, delete_distances, 
                    num_objectives, insert_execution_time, delete_execution_time);
    }
    
    MPI_Finalize();
    return 0;
}
