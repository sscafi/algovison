import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils import ExplanationWindow

class SearchVisualizer:
    def __init__(self, parent):
        self.parent = parent
        self.text = "ABACADABRACADABRA"
        self.pattern = "ABRA"
        self.speed = tk.DoubleVar(value=0.5)
        self.setup_ui()
        
    def setup_ui(self):
        # Control panel
        control_frame = ttk.LabelFrame(self.parent, text="Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Text input
        ttk.Label(control_frame, text="Text:").grid(row=0, column=0)
        self.text_entry = ttk.Entry(control_frame, width=30)
        self.text_entry.insert(0, self.text)
        self.text_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(control_frame, text="Pattern:").grid(row=0, column=2)
        self.pattern_entry = ttk.Entry(control_frame, width=15)
        self.pattern_entry.insert(0, self.pattern)
        self.pattern_entry.grid(row=0, column=3, padx=5)
        
        # Buttons
        ttk.Button(control_frame, text="KMP Search", 
                  command=self.kmp_search).grid(row=0, column=4, padx=5)
        
        # Speed control
        ttk.Label(control_frame, text="Speed:").grid(row=1, column=0)
        ttk.Scale(control_frame, from_=0.1, to=2, variable=self.speed, 
                 orient=tk.HORIZONTAL).grid(row=1, column=1, columnspan=4, sticky=tk.EW)
        
        # Visualization area
        vis_frame = ttk.Frame(self.parent)
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Explanation window
        self.explanation = ExplanationWindow(self.parent)
        self.update_plot()
        
    def update_plot(self, highlight_indices=None, message=None):
        self.ax.clear()
        text = self.text_entry.get()
        pattern = self.pattern_entry.get()
        
        # Create bars for each character
        bars = self.ax.bar(range(len(text)), [1]*len(text), color='lightgray')
        
        # Highlight pattern matches
        if highlight_indices:
            for i in highlight_indices:
                bars[i].set_color('red')
        
        # Set labels
        self.ax.set_xticks(range(len(text)))
        self.ax.set_xticklabels(list(text))
        self.ax.set_yticks([])
        self.ax.set_title("Text Search Visualization")
        
        if message:
            self.explanation.add_message(message)
        self.canvas.draw()
        
    def kmp_search(self):
        text = self.text_entry.get()
        pattern = self.pattern_entry.get()
        self.update_plot(message=f"Starting KMP search for '{pattern}' in '{text}'")
        
        # Compute LPS array
        lps = [0] * len(pattern)
        length = 0
        i = 1
        
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length-1]
                else:
                    lps[i] = 0
                    i += 1
        
        # Perform search
        i = j = 0
        while i < len(text):
            self.update_plot(highlight_indices=[i, j + (i-j)], 
                           message=f"Comparing text[{i}]='{text[i]}' with pattern[{j}]='{pattern[j]}'")
            time.sleep(self.speed.get())
            
            if pattern[j] == text[i]:
                i += 1
                j += 1
                
                if j == len(pattern):
                    start = i - j
                    end = i - 1
                    self.update_plot(highlight_indices=range(start, end+1), 
                                   message=f"Pattern found at index {start}")
                    return start
            else:
                if j != 0:
                    j = lps[j-1]
                    self.update_plot(message=f"Mismatch, jumping to lps[{j-1}]={lps[j-1]}")
                    time.sleep(self.speed.get())
                else:
                    i += 1
        
        self.update_plot(message="Pattern not found in text")
        return -1
