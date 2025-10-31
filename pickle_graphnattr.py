import pickle
import networkx as nx

# Change this to your dataset file
file_path = "./networks/graph_spa_500_0.pickle"

with open(file_path, "rb") as f:
    G = pickle.load(f)

print("Graph loaded successfully!")
print("--- Graph Summary ---")

# Print the graph object
print(G)

# Detailed summary
print(f"\nGraph Type: {type(G)}")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Directed graph degree info
if G.is_directed() and G.number_of_nodes() > 0:
    avg_in_degree = sum(d for n, d in G.in_degree()) / G.number_of_nodes()
    avg_out_degree = sum(d for n, d in G.out_degree()) / G.number_of_nodes()
    print(f"Average in-degree: {avg_in_degree:.4f}")
    print(f"Average out-degree: {avg_out_degree:.4f}")

# Print attributes for the first node
first_node = list(G.nodes())[0]
print(f"\nAttributes for first node ({first_node}):")
print(G.nodes[first_node])

# Print all unique attribute keys
all_attrs = set()
for node in G.nodes():
    all_attrs.update(G.nodes[node].keys())
print("\nAvailable attributes in the graph:", all_attrs)