import pandas as pd
import networkx as nx
import pickle
import gzip
import os

print("Starting Pokec dataset conversion...")

# --- 1. Define File Paths ---
raw_dir = 'raw_data'
networks_dir = 'networks'
os.makedirs(networks_dir, exist_ok=True)

profile_file_gz = os.path.join(raw_dir, 'soc-pokec-profiles.txt.gz')
edge_file_gz = os.path.join(raw_dir, 'soc-pokec-relationships.txt.gz')
output_pickle = os.path.join(networks_dir, 'soc-pokec.pickle')
output_edgelist = os.path.join(networks_dir, 'soc-pokec.txt')

# --- 2. Create the .txt Edgelist File (This part worked, no changes) ---
if not os.path.exists(output_edgelist):
    print(f"Creating edgelist: {output_edgelist} ...")
    count = 0
    with gzip.open(edge_file_gz, 'rt', encoding='utf-8') as f_in:
        with open(output_edgelist, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                try:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        u, v = parts[0], parts[1]
                        f_out.write(f"{u} {v}\n")
                        count += 1
                except Exception as e:
                    print(f"Skipping line due to error: {e}")
    print(f"Done. Wrote {count} edges to {output_edgelist}.")
else:
    print(f"Edgelist {output_edgelist} already exists. Skipping.")


# --- 3. Create the .pickle Graph File (Robust Version) ---
print("Creating .pickle file with all attributes...")

# Define all 60 column names from the readme.txt
column_names = [
    'gender', 'region', 
    'AGE', 'body', 'I_am_working_in_field', 
    'spoken_languages', 'hobbies', 'I_most_enjoy_good_food', 'pets', 
    'body_type', 
    'completed_level_of_education', 'relation_to_smoking', 
    'relation_to_alcohol', 'on_pokec_i_am_looking_for', 'relation_to_casual_sex', 'my_partner_should_be', 
    'marital_status', 'children', 'relation_to_children',
    'profession', 'life_style', 'music', 'cars', 'politics', 
    'relationships', 'art_culture', 'hobbies_interests', 'science_technologies'
    , 'education', 'sport', 'movies', 'travelling', 
    'health', 'companies_brands'
]

# Create an undirected graph
G = nx.Graph()

print("Reading profiles and adding nodes (this may take a minute)...")
node_count = 0
# Use 'latin-1' encoding, common for Eastern European languages
try:
    with gzip.open(profile_file_gz, 'rt', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('\t')
            
            # Handle malformed lines. If a line doesn't even have a user_id, skip it.
            if len(parts) < 1 or parts[0] == 'null' or parts[0] == '':
                continue
            
            try:
                user_id = int(parts[0])
            except ValueError:
                continue # Skip rows with invalid user_id
            
            # Create the attribute dictionary for ALL 60 columns
            attributes = {}
            for i in range(len(column_names)):
                col_name = column_names[i]
                if i < len(parts):
                    # Get the value, store 'null' if it's the 'null' string
                    value = parts[i]
                    attributes[col_name] = value
                else:
                    # If line is too short (due to missing data), fill with 'null'
                    attributes[col_name] = 'null'

            # Add the node with all attributes
            G.add_node(user_id, **attributes)
            node_count += 1

except UnicodeDecodeError:
    print("\n--- ERROR ---")
    print("Failed with 'latin-1' encoding. Trying 'utf-8'.")
    # Reset and try with UTF-8 if latin-1 fails
    G = nx.Graph()
    node_count = 0
    with gzip.open(profile_file_gz, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 1 or parts[0] == 'null' or parts[0] == '':
                continue
            try:
                user_id = int(parts[0])
            except ValueError:
                continue
            attributes = {}
            for i in range(len(column_names)):
                col_name = column_names[i]
                if i < len(parts):
                    value = parts[i]
                    attributes[col_name] = value
                else:
                    attributes[col_name] = 'null'
            G.add_node(user_id, **attributes)
            node_count += 1
            
print(f"Added {node_count} nodes.")

# --- 4. Add Edges to the Graph ---
print(f"Reading edges from {edge_file_gz} to build graph...")
edge_count = 0
with gzip.open(edge_file_gz, 'rt', encoding='utf-8') as f:
    for line in f:
        try:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                # Only add edge if both nodes were successfully added from profiles
                if G.has_node(u) and G.has_node(v):
                    G.add_edge(u, v)
                    edge_count += 1
        except Exception:
            pass # Skip malformed edge lines

print(f"Added {edge_count} edges to the graph.")

# --- 5. Save the .pickle File ---
print(f"Saving graph to {output_pickle}...")
with open(output_pickle, 'wb') as f:
    pickle.dump(G, f, protocol=pickle.DEFAULT_PROTOCOL)

print("\n--- Conversion Complete! ---")
print(f"Successfully created:")
print(f"1. {output_pickle} (Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})")
print(f"2. {output_edgelist} (Text edgelist)")