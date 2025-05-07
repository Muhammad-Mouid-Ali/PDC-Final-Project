#include <mpi.h>
#include <metis.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <sstream>
#include <string>
#include <climits>

using namespace std;

// Structure to represent an edge
struct Edge {
    int dest;
    int weight;
};

// Structure to represent a path (for SOSP/MOSP)
struct Path {
    vector<int> nodes;
    int total_weight;
    bool operator<(const Path& other) const {
        return total_weight < other.total_weight;
    }
};

// Function to read the graph (Master only)
map<int, vector<Edge>> read_graph(const string& filename, int& num_vertices, int& num_edges) {
    map<int, vector<Edge>> graph;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    string line;
    num_edges = 0;
    num_vertices = 0;
    while (getline(file, line)) {
        istringstream iss(line);
        int src, dest, weight;
        if (iss >> src >> dest >> weight) {
            if (src < 0 || dest < 0) continue; // Skip invalid vertices
            graph[src].push_back({dest, weight});
            graph[dest]; // Ensure dest vertex exists
            num_vertices = max({num_vertices, src + 1, dest + 1});
            num_edges++;
        }
    }
    file.close();
    if (num_vertices == 0) {
        cerr << "Empty graph detected" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return graph;
}

// Convert graph to METIS format
void graph_to_metis(const map<int, vector<Edge>>& graph, int num_vertices, vector<idx_t>& xadj, vector<idx_t>& adjncy, vector<idx_t>& adjwgt) {
    xadj.clear();
    adjncy.clear();
    adjwgt.clear();
    xadj.push_back(0);
    for (int v = 0; v < num_vertices; ++v) {
        auto it = graph.find(v);
        if (it != graph.end()) {
            for (const auto& edge : it->second) {
                adjncy.push_back(edge.dest);
                adjwgt.push_back(edge.weight);
            }
        }
        xadj.push_back(adjncy.size());
    }
}

// Partition graph using METIS
void partition_graph(int num_vertices, int num_parts, vector<idx_t>& xadj, vector<idx_t>& adjncy, vector<idx_t>& adjwgt, vector<idx_t>& part) {
    if (num_vertices == 0 || xadj.empty()) {
        cerr << "Invalid graph for METIS partitioning" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    idx_t ncon = 1;
    idx_t objval;
    part.resize(num_vertices);
    vector<idx_t> xadj_copy = xadj;
    vector<idx_t> adjncy_copy = adjncy;
    vector<idx_t> adjwgt_copy = adjwgt;
    int ret = METIS_PartGraphKway(&num_vertices, &ncon, xadj_copy.data(), adjncy_copy.data(), nullptr, nullptr, adjwgt_copy.data(),
                                  &num_parts, nullptr, nullptr, nullptr, &objval, part.data());
    if (ret != METIS_OK) {
        cerr << "METIS partitioning failed with code: " << ret << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Distribute partitions to MPI processes
void distribute_partitions(const map<int, vector<Edge>>& full_graph, const vector<idx_t>& part, int rank, int size, map<int, vector<Edge>>& local_graph, vector<int>& local_vertices) {
    local_graph.clear();
    local_vertices.clear();
    for (const auto& [v, edges] : full_graph) {
        if (v < static_cast<int>(part.size()) && part[v] == rank) {
            local_graph[v] = edges;
            local_vertices.push_back(v);
        }
    }
}

// Compute partial SOSP on local partition (simplified Dijkstra-like)
vector<Path> compute_sosp(const map<int, vector<Edge>>& local_graph, const vector<int>& local_vertices) {
    vector<Path> sosp;
    for (int src : local_vertices) {
        map<int, int> dist;
        map<int, vector<int>> pred;
        set<pair<int, int>> pq; // {distance, vertex}
        for (int v : local_vertices) dist[v] = INT_MAX;
        dist[src] = 0;
        pq.insert({0, src});
        while (!pq.empty()) {
            int d = pq.begin()->first;
            int u = pq.begin()->second;
            pq.erase(pq.begin());
            if (d > dist[u]) continue;
            auto it = local_graph.find(u);
            if (it != local_graph.end()) {
                for (const auto& edge : it->second) {
                    int v = edge.dest;
                    int w = edge.weight;
                    if (dist.count(v) == 0) dist[v] = INT_MAX; // Initialize if not present
                    if (dist[u] + w < dist[v]) {
                        dist[v] = dist[u] + w;
                        pred[v] = {u};
                        pq.insert({dist[v], v});
                    }
                }
            }
        }
        for (const auto& [dest, d] : dist) {
            if (d != INT_MAX && dest != src) {
                Path p;
                p.total_weight = d;
                int curr = dest;
                while (curr != src) {
                    p.nodes.push_back(curr);
                    curr = pred[curr][0];
                }
                p.nodes.push_back(src);
                reverse(p.nodes.begin(), p.nodes.end());
                sosp.push_back(p);
            }
        }
    }
    return sosp;
}

// Exchange boundary info (simplified)
vector<Path> exchange_boundary_info(const vector<Path>& local_sosp, int rank, int size) {
    vector<Path> all_sosp = local_sosp;
    for (int src_rank = 0; src_rank < size; ++src_rank) {
        int len;
        if (rank == src_rank) {
            len = local_sosp.size();
        }
        MPI_Bcast(&len, 1, MPI_INT, src_rank, MPI_COMM_WORLD);
        vector<int> all_nodes;
        vector<int> all_weights(len);
        if (rank == src_rank) {
            for (const auto& path : local_sosp) {
                all_nodes.insert(all_nodes.end(), path.nodes.begin(), path.nodes.end());
                all_weights.push_back(path.total_weight);
            }
        }
        int total_nodes;
        if (rank == src_rank) {
            total_nodes = all_nodes.size();
        }
        MPI_Bcast(&total_nodes, 1, MPI_INT, src_rank, MPI_COMM_WORLD);
        all_nodes.resize(total_nodes);
        MPI_Bcast(all_nodes.data(), total_nodes, MPI_INT, src_rank, MPI_COMM_WORLD);
        MPI_Bcast(all_weights.data(), len, MPI_INT, src_rank, MPI_COMM_WORLD);
        if (rank != src_rank) {
            int node_idx = 0;
            for (int i = 0; i < len; ++i) {
                Path p;
                int nodes_len;
                if (i == 0) {
                    nodes_len = total_nodes / len;
                } else {
                    nodes_len = all_nodes[node_idx - 1] + 1;
                }
                p.nodes.resize(nodes_len);
                for (int j = 0; j < nodes_len; ++j) {
                    p.nodes[j] = all_nodes[node_idx++];
                }
                p.total_weight = all_weights[i];
                all_sosp.push_back(p);
            }
        }
    }
    return all_sosp;
}

// Merge partial SOSPs into global SOSP
vector<Path> merge_sosp(const vector<Path>& all_sosp) {
    return all_sosp; // Simplified: return all paths
}

// Compute MOSP (simplified)
vector<Path> compute_mosp(const vector<Path>& global_sosp) {
    return global_sosp; // Simplified: return global_sosp as MOSP
}

// Pareto optimality check
vector<Path> compute_pareto_set(const vector<Path>& mosp) {
    vector<Path> pareto_set;
    for (const auto& p : mosp) {
        bool dominated = false;
        for (const auto& q : mosp) {
            if (q.total_weight < p.total_weight) {
                dominated = true;
                break;
            }
        }
        if (!dominated) pareto_set.push_back(p);
    }
    return pareto_set;
}

// Count edges in graph
int count_edges(const map<int, vector<Edge>>& graph) {
    int edges = 0;
    for (const auto& [v, e] : graph) edges += e.size();
    return edges;
}

// Broadcast graph to all processes
void broadcast_graph(map<int, vector<Edge>>& full_graph, int& num_vertices, int& num_edges, int rank) {
    MPI_Bcast(&num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    vector<int> serialized;
    if (rank == 0) {
        for (const auto& [v, edges] : full_graph) {
            for (const auto& e : edges) {
                serialized.push_back(v);
                serialized.push_back(e.dest);
                serialized.push_back(e.weight);
            }
        }
    }
    int len = serialized.size();
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) serialized.resize(len);
    MPI_Bcast(serialized.data(), len, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        full_graph.clear();
        for (int i = 0; i < len; i += 3) {
            int src = serialized[i];
            int dest = serialized[i + 1];
            int weight = serialized[i + 2];
            full_graph[src].push_back({dest, weight});
            full_graph[dest]; // Ensure dest exists
        }
    }
}

// User menu for graph updates
void user_menu(map<int, vector<Edge>>& full_graph, int& num_vertices, int& num_edges, int rank) {
    if (rank != 0) return;
    while (true) {
        cout << "\nUser Menu:\n1. Insert node\n2. Delete node\n3. Insert edge\n4. Delete edge\n5. Exit\nEnter choice: ";
        int choice;
        cin >> choice;
        if (choice == 5) break;
        if (choice == 1) {
            int new_node = num_vertices++;
            full_graph[new_node] = {};
            cout << "Node " << new_node << " added.\n";
        } else if (choice == 2) {
            int node;
            cout << "Enter node to delete: ";
            cin >> node;
            if (full_graph.erase(node)) {
                for (auto& [v, edges] : full_graph) {
                    edges.erase(remove_if(edges.begin(), edges.end(),
                                          [node](const Edge& e) { return e.dest == node; }), edges.end());
                }
                num_edges = count_edges(full_graph);
                cout << "Node " << node << " deleted.\n";
            }
        } else if (choice == 3) {
            int src, dest, weight;
            cout << "Enter source, destination, weight: ";
            cin >> src >> dest >> weight;
            full_graph[src].push_back({dest, weight});
            full_graph[dest]; // Ensure dest exists
            num_edges++;
            cout << "Edge " << src << "->" << dest << " added.\n";
        } else if (choice == 4) {
            int src, dest;
            cout << "Enter source, destination: ";
            cin >> src >> dest;
            auto& edges = full_graph[src];
            edges.erase(remove_if(edges.begin(), edges.end(),
                                  [dest](const Edge& e) { return e.dest == dest; }), edges.end());
            num_edges = count_edges(full_graph);
            cout << "Edge " << src << "->" << dest << " deleted.\n";
        }
        broadcast_graph(full_graph, num_vertices, num_edges, rank);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    map<int, vector<Edge>> full_graph, local_graph;
    int num_vertices = 0, num_edges = 0;
    vector<int> local_vertices;

    // Step 1: Read graph (Master only)
    if (rank == 0) {
        full_graph = read_graph("weightedfacebook_graph.txt", num_vertices, num_edges);
    }
    broadcast_graph(full_graph, num_vertices, num_edges, rank);

    // Step 2: Partition graph using METIS
    vector<idx_t> xadj, adjncy, adjwgt, part;
    if (rank == 0) {
        graph_to_metis(full_graph, num_vertices, xadj, adjncy, adjwgt);
        partition_graph(num_vertices, size, xadj, adjncy, adjwgt, part);
    }
    part.resize(num_vertices);
    MPI_Bcast(part.data(), num_vertices, MPI_INT, 0, MPI_COMM_WORLD);
    distribute_partitions(full_graph, part, rank, size, local_graph, local_vertices);

    // Step 3: Compute SOSP
    vector<Path> local_sosp = compute_sosp(local_graph, local_vertices);
    vector<Path> all_sosp = exchange_boundary_info(local_sosp, rank, size);
    vector<Path> global_sosp = merge_sosp(all_sosp);

    // Step 4 & 5: Compute MOSP
    vector<Path> mosp = compute_mosp(global_sosp);

    // Step 6: Pareto optimality
    vector<Path> pareto_set = compute_pareto_set(mosp);

    // Output Pareto optimal paths
    if (rank == 0) {
        cout << "Pareto Optimal Paths:\n";
        for (const auto& p : pareto_set) {
            cout << "Path: ";
            for (int v : p.nodes) cout << v << " ";
            cout << "Weight: " << p.total_weight << endl;
        }
    }

    // Step 7: User menu
    user_menu(full_graph, num_vertices, num_edges, rank);

    MPI_Finalize();
    return 0;
}