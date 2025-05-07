#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <climits>
#include <chrono>
#include <map>

using namespace std;

struct Edge {
    int src, dest;
    vector<int> weights;
    Edge(int s, int d, const vector<int>& w) : src(s), dest(d), weights(w) {}
};

class Graph {
public:
    int V;
    int numObjectives;
    vector<vector<Edge>> adj;

    Graph(int vertices = 0, int objectives = 2) : V(vertices), numObjectives(objectives) {
        adj.resize(V);
    }

    bool addEdge(int src, int dest, const vector<int>& weights) {
        if (src >= V || dest >= V) return false;
        adj[src].emplace_back(src, dest, weights);
        return true;
    }

    bool removeEdge(int src, int dest) {
        if (src >= V || dest >= V) return false;
        auto& edges = adj[src];
        size_t before = edges.size();
        edges.erase(remove_if(edges.begin(), edges.end(),
            [dest](const Edge& e) { return e.dest == dest; }), edges.end());
        return edges.size() < before;
    }

    bool readGraph(const string& filename) {
        ifstream infile(filename);
        if (!infile.is_open()) return false;
        string line;
        int maxNode = -1;
        vector<Edge> tempEdges;
        while (getline(infile, line)) {
            istringstream iss(line);
            int src, dest;
            vector<int> weights;
            iss >> src >> dest;
            int w;
            while (iss >> w) weights.push_back(w);
            if (weights.size() > numObjectives) numObjectives = weights.size();
            maxNode = max({ maxNode, src, dest });
            tempEdges.emplace_back(src, dest, weights);
        }
        infile.close();
        V = maxNode + 1;
        adj.assign(V, vector<Edge>());
        for (const auto& e : tempEdges) {
            adj[e.src].push_back(e);
        }
        return true;
    }
};

struct SOSP_Tree {
    vector<int> distance;
    vector<int> parent;
    SOSP_Tree(int V) : distance(V, INT_MAX), parent(V, -1) {}
};

SOSP_Tree dijkstra(const Graph& G, int src, int objective) {
    SOSP_Tree tree(G.V);
    tree.distance[src] = 0;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
    pq.push({ 0, src });
    while (!pq.empty()) {
        int d = pq.top().first, u = pq.top().second; pq.pop();
        if (d > tree.distance[u]) continue;
        for (const auto& edge : G.adj[u]) {
            int v = edge.dest;
            int weight = (objective < edge.weights.size()) ? edge.weights[objective] : edge.weights[0];
            if (tree.distance[u] + weight < tree.distance[v]) {
                tree.distance[v] = tree.distance[u] + weight;
                tree.parent[v] = u;
                pq.push({ tree.distance[v], v });
            }
        }
    }
    return tree;
}

Graph buildEnsembleGraph(const vector<SOSP_Tree>& trees, const Graph& G) {
    Graph ensemble(G.V, G.numObjectives);
    map<pair<int, int>, int> count;

    for (const auto& tree : trees) {
        for (int v = 0; v < G.V; v++) {
            int u = tree.parent[v];
            if (u != -1) count[{u, v}]++;
        }
    }

    for (const auto& kv : count) {
        int u = kv.first.first;
        int v = kv.first.second;
        int freq = kv.second;
        vector<int> weight = { G.numObjectives - freq + 1 };
        ensemble.addEdge(u, v, weight);
    }
    return ensemble;
}

vector<int> reconstructPath(const SOSP_Tree& tree, int dest) {
    vector<int> path;
    for (int v = dest; v != -1; v = tree.parent[v]) path.push_back(v);
    reverse(path.begin(), path.end());
    return path;
}

void computeMOSP(Graph& G, int src, int dest) {
    auto start = chrono::high_resolution_clock::now();

    vector<SOSP_Tree> trees;
    for (int i = 0; i < G.numObjectives; i++) {
        trees.push_back(dijkstra(G, src, i));
    }

    Graph ensemble = buildEnsembleGraph(trees, G);
    SOSP_Tree mospTree = dijkstra(ensemble, src, 0);
    vector<int> path = reconstructPath(mospTree, dest);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    vector<int> totalDistances(G.numObjectives, 0);
    for (size_t i = 0; i < path.size() - 1; i++) {
        int u = path[i];
        int v = path[i + 1];
        for (const auto& edge : G.adj[u]) {
            if (edge.dest == v) {
                for (int j = 0; j < G.numObjectives; j++) {
                    totalDistances[j] += (j < edge.weights.size()) ? edge.weights[j] : edge.weights[0];
                }
                break;
            }
        }
    }

    // Console output
    cout << "MOSP Path: ";
    if (path.size() > 10) {
        for (int i = 0; i < 5; i++) cout << path[i] << " ";
        cout << "... ";
        for (int i = path.size() - 5; i < path.size(); i++) cout << path[i] << " ";
    } else {
        for (int v : path) cout << v << " ";
    }
    cout << endl;

    cout << "Distances for each objective:" << endl;
    for (int i = 0; i < G.numObjectives; i++) {
        cout << "Objective " << i + 1 << ": " << totalDistances[i] << endl;
    }
    cout << "Execution time: " << duration.count() << " ms" << endl;

    // Save to file
    ofstream out("mosp_result.txt");
    out << "MOSP Path: ";
    for (int v : path) out << v << " ";
    out << endl;
    for (int i = 0; i < G.numObjectives; i++) {
        out << "Objective " << i + 1 << ": " << totalDistances[i] << endl;
    }
    out << "Execution time: " << duration.count() << " ms" << endl;
    out.close();
}

void menu(Graph& G) {
    int src = 0, dest = G.V - 1;
    computeMOSP(G, src, dest);

    int choice;
    do {
        cout << "\n1. Add Edge\n2. Delete Edge\n3. Exit\nChoice: ";
        cin >> choice;
        if (choice == 1) {
            int e_src, e_dest;
            vector<int> weights;
            cout << "Enter src dest followed by " << G.numObjectives << " weights: ";
            cin >> e_src >> e_dest;
            for (int i = 0; i < G.numObjectives; i++) {
                int w;
                cin >> w;
                weights.push_back(w);
            }
            if (G.addEdge(e_src, e_dest, weights)) {
                cout << "Edge added. Recomputing MOSP...\n";
                computeMOSP(G, src, dest);
            }
        } else if (choice == 2) {
            int e_src, e_dest;
            cout << "Enter src and dest to remove edge: ";
            cin >> e_src >> e_dest;
            if (G.removeEdge(e_src, e_dest)) {
                cout << "Edge removed. Recomputing MOSP...\n";
                computeMOSP(G, src, dest);
            } else {
                cout << "Edge not found.\n";
            }
        }
    } while (choice != 3);
}

int main() {
    Graph G;
    string filename = "weightedfacebook_graph.txt";
    if (!G.readGraph(filename)) {
        cerr << "Error reading graph file.\n";
        return 1;
    }

    menu(G);
    return 0;
}
