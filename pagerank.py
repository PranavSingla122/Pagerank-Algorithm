import pandas as pd
import networkx as nx

data = pd.read_csv('data1.csv')

# Initialize a directed graph
G = nx.DiGraph()

# Add edges to the graph
for _, row in data.iterrows():
    person = row[0]
    impressive_people = row[1:].dropna().tolist()
    for impressive_person in impressive_people:
        G.add_edge(person, impressive_person)

# Random Walk Algorithm (PageRank)
pagerank_scores = nx.pagerank(G, alpha=0.85)  # alpha is the damping factor
pagerank_top10 = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]

# Equal Points Distribution Algorithm (modified to include damping factor)
def equal_points_distribution(G, alpha=0.85, iterations=100):
    nodes = G.nodes()
    num_nodes = len(nodes)
    points = {node: 1.0 / num_nodes for node in nodes}
    dangling_nodes = [node for node in nodes if len(G.out_edges(node)) == 0]

    for _ in range(iterations):
        new_points = {node: (1 - alpha) / num_nodes for node in nodes}
        for node in nodes:
            out_edges = list(G.out_edges(node))
            if out_edges:
                distributed_points = alpha * points[node] / len(out_edges)
                for _, target in out_edges:
                    new_points[target] += distributed_points
            else:
                for target in nodes:
                    new_points[target] += alpha * points[node] / num_nodes
        points = new_points
    return points

epd_scores = equal_points_distribution(G)
epd_top10 = sorted(epd_scores.items(), key=lambda x: x[1], reverse=True)[:10]

# Extract the top 10 persons from both algorithms
pagerank_top10_persons = [person for person, score in pagerank_top10]
epd_top10_persons = [person for person, score in epd_top10]

# Find the number of common people in both lists
common_people = set(pagerank_top10_persons).intersection(set(epd_top10_persons))
num_common_people = len(common_people)

# Display the top 10 important persons from both algorithms
print("Top 10 persons based on Random Walk (PageRank):")
for person, score in pagerank_top10:
    print(f"{person}: {score}")

print("\nTop 10 persons based on Equal Points Distribution:")
for person, score in epd_top10:
    print(f"{person}: {score}")

print(f"\nNumber of common people in both top 10 lists: {num_common_people}")
print(f"Common people: {common_people}")
