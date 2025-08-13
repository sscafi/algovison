import tkinter as tk
from tkinter import ttk, scrolledtext
from sorting import SortingVisualizer
from search import SearchVisualizer
from graph import GraphVisualizer
from styles import configure_styles

class AlgorithmVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Algorithm Visualizer")
        self.root.geometry("1000x700")
        configure_styles()
        self.setup_ui()
        
    def setup_ui(self):
        # Create main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create frames for each category
        self.sorting_frame = ttk.Frame(self.notebook)
        self.search_frame = ttk.Frame(self.notebook)
        self.graph_frame = ttk.Frame(self.notebook)
        
        # Add tabs
        self.notebook.add(self.sorting_frame, text="Sorting Algorithms")
        self.notebook.add(self.search_frame, text="Search Algorithms")
        self.notebook.add(self.graph_frame, text="Graph Algorithms")
        
        # Initialize visualizers
        SortingVisualizer(self.sorting_frame)
        SearchVisualizer(self.search_frame)
        GraphVisualizer(self.graph_frame)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X)

if __name__ == "__main__":
    root = tk.Tk()
    app = AlgorithmVisualizer(root)
    root.mainloop()
