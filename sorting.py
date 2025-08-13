import random
import time
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils import ExplanationWindow

class SortingVisualizer:
    def __init__(self, parent):
        self.parent = parent
        self.array = []
        self.speed = tk.DoubleVar(value=0.5)
        self.setup_ui()
        
    def setup_ui(self):
        # Control panel
        control_frame = ttk.LabelFrame(self.parent, text="Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Buttons
        ttk.Button(control_frame, text="Generate Random Array", 
                  command=self.generate_array).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Bubble Sort", 
                  command=lambda: self.run_algorithm("bubble")).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Insertion Sort", 
                  command=lambda: self.run_algorithm("insertion")).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Quick Sort", 
                  command=lambda: self.run_algorithm("quick")).grid(row=0, column=3, padx=5)
        ttk.Button(control_frame, text="Merge Sort", 
                  command=lambda: self.run_algorithm("merge")).grid(row=0, column=4, padx=5)
        
        # Speed control
        ttk.Label(control_frame, text="Speed:").grid(row=1, column=0)
        ttk.Scale(control_frame, from_=0.1, to=2, variable=self.speed, 
                 orient=tk.HORIZONTAL).grid(row=1, column=1, columnspan=4, sticky=tk.EW)
        
        # Visualization area
        vis_frame = ttk.Frame(self.parent)
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Explanation window
        self.explanation = ExplanationWindow(self.parent)
        
    def generate_array(self, size=15):
        self.array = [random.randint(1, 100) for _ in range(size)]
        self.update_plot("Generated new random array")
        
    def update_plot(self, message=None):
        self.ax.clear()
        bars = self.ax.bar(range(len(self.array)), self.array, color='skyblue')
        
        # Customize plot
        self.ax.set_title("Sorting Visualization")
        self.ax.set_xlabel("Index")
        self.ax.set_ylabel("Value")
        self.ax.set_ylim(0, max(self.array) + 10)
        
        if message:
            self.explanation.add_message(message)
        self.canvas.draw()
        
    def run_algorithm(self, algorithm):
        if not self.array:
            self.generate_array()
            
        if algorithm == "bubble":
            self.bubble_sort()
        elif algorithm == "insertion":
            self.insertion_sort()
        elif algorithm == "quick":
            self.quick_sort_wrapper()
        elif algorithm == "merge":
            self.merge_sort_wrapper()
    
    def bubble_sort(self):
        n = len(self.array)
        for i in range(n):
            for j in range(0, n-i-1):
                if self.array[j] > self.array[j+1]:
                    self.array[j], self.array[j+1] = self.array[j+1], self.array[j]
                    self.update_plot(f"Swapped {self.array[j]} and {self.array[j+1]}")
                    time.sleep(self.speed.get())
    
    def insertion_sort(self):
        for i in range(1, len(self.array)):
            key = self.array[i]
            j = i-1
            while j >= 0 and key < self.array[j]:
                self.array[j+1] = self.array[j]
                self.update_plot(f"Moving {self.array[j]} to position {j+1}")
                time.sleep(self.speed.get())
                j -= 1
            self.array[j+1] = key
            self.update_plot(f"Inserted {key} at position {j+1}")
            time.sleep(self.speed.get())
    
    def quick_sort_wrapper(self):
        self.quick_sort(0, len(self.array)-1)
    
    def quick_sort(self, low, high):
        if low < high:
            pi = self.partition(low, high)
            self.quick_sort(low, pi-1)
            self.quick_sort(pi+1, high)
    
    def partition(self, low, high):
        pivot = self.array[high]
        i = low - 1
        
        for j in range(low, high):
            if self.array[j] < pivot:
                i += 1
                self.array[i], self.array[j] = self.array[j], self.array[i]
                self.update_plot(f"Swapped {self.array[i]} and {self.array[j]}")
                time.sleep(self.speed.get())
        
        self.array[i+1], self.array[high] = self.array[high], self.array[i+1]
        self.update_plot(f"Placed pivot {pivot} at position {i+1}")
        time.sleep(self.speed.get())
        return i + 1
    
    def merge_sort_wrapper(self):
        self.merge_sort(0, len(self.array)-1)
    
    def merge_sort(self, l, r):
        if l < r:
            m = (l + r) // 2
            self.merge_sort(l, m)
            self.merge_sort(m+1, r)
            self.merge(l, m, r)
    
    def merge(self, l, m, r):
        n1 = m - l + 1
        n2 = r - m
        
        L = self.array[l:m+1]
        R = self.array[m+1:r+1]
        
        i = j = 0
        k = l
        
        while i < n1 and j < n2:
            if L[i] <= R[j]:
                self.array[k] = L[i]
                i += 1
            else:
                self.array[k] = R[j]
                j += 1
            self.update_plot(f"Merging elements at position {k}")
            time.sleep(self.speed.get())
            k += 1
        
        while i < n1:
            self.array[k] = L[i]
            i += 1
            k += 1
            self.update_plot(f"Copying remaining left element at {k-1}")
            time.sleep(self.speed.get())
        
        while j < n2:
            self.array[k] = R[j]
            j += 1
            k += 1
            self.update_plot(f"Copying remaining right element at {k-1}")
            time.sleep(self.speed.get())
