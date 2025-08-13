import tkinter as tk
from tkinter import ttk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils import ExplanationWindow
import heapq

class GraphVisualizer:
    def __init__(self, parent):
        self.parent = parent
        self.graph = {
            'A': {'B': 4, 'H': 8},
            'B': {'A': 4, 'C': 8, 'H': 11},
            'C': {'B': 8, 'D': 7, 'F': 4, 'I': 2},
            'D': {'C': 7, 'E': 9, 'F': 14},
            'E': {'D': 9, 'F': 10},
            'F': {'C': 4, 'D': 14, 'E': 10, 'G': 2},
            'G': {'F': 2, 'H': 1, 'I': 6},
            'H': {'A': 8, 'B': 11, 'G': 1, 'I': 7},
            'I': {'C': 2, 'G': 6, 'H': 7}
        }
        self.speed = tk.DoubleVar(value=0.5)
        self.setup_ui()
        
    def setup_ui(self):
        # Control panel
        control_frame = ttk.LabelFrame(self.parent, text="Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Buttons
        ttk.Button(control_frame, text="Dijkstra's", 
                  command=lambda: self.run_algorithm("dijkstra")).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Prim's", 
                  command=lambda: self.run_algorithm("prim")).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Kruskal's", 
                  command=lambda: self.run_algorithm("kruskal")).grid(row=0, column=2, padx=5)
        
        # Start node input
        ttk.Label(control_frame, text="Start Node:").grid(row=0, column=3)
        self.start_node = ttk.Combobox(control_frame, values=list(self.graph.keys()), width=3)
        self.start_node.current(0)
        self.start_node.grid(row=0, column=4, padx=5)
        
        # Speed control
        ttk.Label(control_frame, text="Speed:").grid(row=1, column=0)
        ttk.Scale(control_frame, from_=0.1, to=2, variable=self.speed, 
                 orient=tk.HORIZONTAL).grid(row=1, column=1, columnspan=4, sticky=tk.EW)
        
        # Visualization area
        vis_frame = ttk.Frame(self.parent)
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Explanation window
        self.explanation = ExplanationWindow(self.parent)
        self.draw_graph()
        
    def draw_graph(self, highlighted_edges=None, node_colors=None, message=None):
        self.ax.clear()
        G = nx.Graph()
        
        # Add nodes and edges
        for node in self.graph:
            G.add_node(node)
            for neighbor, weight in self.graph[node].items():
                G.add_edge(node, neighbor, weight=weight)
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G)
        
        # Default colors
        if node_colors is None:
            node_colors = ['skyblue'] * len(G.nodes())
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, ax=self.ax)
        nx.draw_networkx_labels(G, pos, ax=self.ax)
        
        # Draw all edges in gray
        nx.draw_networkx_edges(G, pos, edge_color='gray', ax=self.ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), ax=self.ax)
        
        # Highlight specific edges if provided
        if highlighted_edges:
            nx.draw_networkx_edges(G, pos, edgelist=highlighted_edges, 
                                 edge_color='red', width=2, ax=self.ax)
        
        self.ax.set_title("Graph Algorithm Visualization")
        self.ax.axis('off')
        
        if message:
            self.explanation.add_message(message)
        self.canvas.draw()
        
    def run_algorithm(self, algorithm):
        if algorithm == "dijkstra":
            self.dijkstra(self.start_node.get())
        elif algorithm == "prim":
            self.prim()
        elif algorithm == "kruskal":
            self.kruskal()
    
    def dijkstra(self, start):
        distances = {node: float('inf') for node in self.graph}
        distances[start] = 0
        heap = [(0, start)]
        visited = set()
        
        while heap:
            current_dist, current_node = heapq.heappop(heap)
            
            if current_node in visited:
                continue
                
            visited.add(current_node)
            self.draw_graph(
                node_colors=['red' if n == current_node else 'skyblue' for n in self.graph],
                message=f"Visiting node {current_node} with distance {current_dist}"
            )
            time.sleep(self.speed.get())
            
            for neighbor, weight in self.graph[current_node].items():
                distance = current_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(heap, (distance, neighbor))
                    self.draw_graph(
                        highlighted_edges=[(current_node, neighbor)],
                        message=f"Updating distance to {neighbor} to {distance}"
                    )
                    time.sleep(self.speed.get())
        
        self.explanation.add_message(f"Final distances from {start}: {distances}")
    
    def prim(self):
        start = next(iter(self.graph))
        visited = {start}
        edges = [
            (weight, start, neighbor)
            for neighbor, weight in self.graph[start].items()
        ]
        heapq.heapify(edges)
        mst_edges = []
        
        while edges:
            weight, u, v = heapq.heappop(edges)
            if v not in visited:
                visited.add(v)
                mst_edges.append((u, v))
                self.draw_graph(
                    highlighted_edges=mst_edges,
                    node_colors=['red' if n in visited else 'skyblue' for n in self.graph],
                    message=f"Added edge {u}-{v} with weight {weight} to MST"
                )
                time.sleep(self.speed.get())
                
                for neighbor, weight in self.graph[v].items():
                    if neighbor not in visited:
                        heapq.heappush(edges, (weight, v, neighbor))
        
        self.explanation.add_message(f"MST edges: {mst_edges}")
    
    def kruskal(self):
        edges = []
        for u in self.graph:
            for v, weight in self.graph[u].items():
                edges.append((weight, u, v))
        edges.sort()
        
        parent = {node: node for node in self.graph}
        rank = {node: 0 for node in self.graph}
        mst_edges = []
        
        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]
        
        for weight, u, v in edges:
            root_u = find(u)
            root_v = find(v)
            
            if root_u != root_v:
                mst_edges.append((u, v))
                if rank[root_u] > rank[root_v]:
                    parent[root_v] = root_u
                else:
                    parent[root_u] = root_v
                    if rank[root_u] == rank[root_v]:
                        rank[root_v] += 1
                
                self.draw_graph(
                    highlighted_edges=mst_edges,
                    message=f"Added edge {u}-{v} with weight {weight} to MST"
                )
                time.sleep(self.speed.get())
        
        self.explanation.add_message(f"MST edges: {mst_edges}")
