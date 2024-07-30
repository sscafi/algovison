import random
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq

# Global variable to track if the GUI is active
gui_active = True

# Example sorting algorithms
def bubble_sort(arr, draw, pause, explain):
    draw(arr)
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                draw(arr)
                explain(f'Swapping {arr[j]} and {arr[j+1]}')
                time.sleep(pause)
                if not gui_active:  # Check if GUI is active
                    return  # Exit if GUI is closed

def insertion_sort(arr, draw, pause, explain):
    draw(arr)
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            draw(arr)
            explain(f'Moving {arr[j]} to the right')
            time.sleep(pause)
            if not gui_active:  # Check if GUI is active
                return  # Exit if GUI is closed
            j -= 1
        arr[j + 1] = key
        draw(arr)
        explain(f'Inserting {key} at position {j + 1}')
        time.sleep(pause)
        if not gui_active:  # Check if GUI is active
            return  # Exit if GUI is closed

def quick_sort(arr, low, high, draw, pause, explain):
    if low < high:
        pi = partition(arr, low, high, draw, pause, explain)
        quick_sort(arr, low, pi-1, draw, pause, explain)
        quick_sort(arr, pi+1, high, draw, pause, explain)

def partition(arr, low, high, draw, pause, explain):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            draw(arr)
            explain(f'Swapping {arr[i]} and {arr[j]}')
            time.sleep(pause)
            if not gui_active:  # Check if GUI is active
                return  # Exit if GUI is closed
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    draw(arr)
    explain(f'Swapping pivot {pivot} with {arr[i + 1]}')
    time.sleep(pause)
    return i + 1

def merge_sort(arr, draw, pause, explain):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L, draw, pause, explain)
        merge_sort(R, draw, pause, explain)

        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                explain(f'Adding {L[i]} to merged array')
                i += 1
            else:
                arr[k] = R[j]
                explain(f'Adding {R[j]} to merged array')
                j += 1
            draw(arr)
            time.sleep(pause)
            if not gui_active:  # Check if GUI is active
                return  # Exit if GUI is closed
            k += 1

        while i < len(L):
            arr[k] = L[i]
            explain(f'Adding {L[i]} to merged array')
            i += 1
            k += 1
            draw(arr)
            time.sleep(pause)
            if not gui_active:  # Check if GUI is active
                return  # Exit if GUI is closed

        while j < len(R):
            arr[k] = R[j]
            explain(f'Adding {R[j]} to merged array')
            j += 1
            k += 1
            draw(arr)
            time.sleep(pause)
            if not gui_active:  # Check if GUI is active
                return  # Exit if GUI is closed

# KMP Search Algorithm
def kmp_search(text, pattern, explain):
    m = len(pattern)
    n = len(text)

    # Create lps array
    lps = [0] * m
    j = 0  # length of previous longest prefix suffix
    compute_lps_array(pattern, m, lps)

    i = 0  # index for text
    while n - i >= m:
        if pattern[j] == text[i]:
            explain(f'Matching {pattern[j]} with {text[i]}')
            i += 1
            j += 1

        if j == m:
            explain(f'Pattern found at index {i - j}')
            return i - j  # Found the pattern
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
                explain(f'Pattern mismatch, jumping to index {j}')
            else:
                i += 1

    explain('Pattern not found')
    return -1  # Pattern not found

def compute_lps_array(pattern, m, lps):
    length = 0  # length of the previous longest prefix suffix
    lps[0] = 0  # lps[0] is always 0
    i = 1

    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

def visualize(text, title="Visualization"):
    plt.clf()
    arr = list(text)
    plt.bar(range(len(arr)), arr, color='skyblue')
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Character')
    plt.xticks(range(len(arr)), arr)
    plt.ylim(0, 255)  # Characters range
    plt.show(block=False)  # Non-blocking show for Matplotlib

# Visualization function for sorting algorithms
def draw(arr):
    plt.clf()
    plt.bar(range(len(arr)), arr, color='cornflowerblue')
    plt.title('Sorting Algorithm Visualization')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.ylim(0, max(arr) + 10)
    plt.pause(0.1)

# Dijkstra's Algorithm
def dijkstra(graph, start, explain):
    explain(f'Starting Dijkstra\'s Algorithm from {start}')
    queue = []
    heapq.heappush(queue, (0, start))
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    while queue:
        current_distance, current_node = heapq.heappop(queue)
        explain(f'Visiting node {current_node} with distance {current_distance}')

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
                explain(f'Updating distance of {neighbor} to {distance}')

    return distances

# Prim's Algorithm
def prim(graph, explain):
    start = next(iter(graph))
    visited = {start}
    edges = [(cost, start, to) for to, cost in graph[start].items()]
    heapq.heapify(edges)
    mst = []
    explain(f'Starting Prim\'s Algorithm from {start}')

    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst.append((frm, to, cost))
            explain(f'Adding edge {frm}-{to} with cost {cost}')
            for to_next, cost in graph[to].items():
                if to_next not in visited:
                    heapq.heappush(edges, (cost, to, to_next))
    
    return mst

# Kruskal's Algorithm
def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def kruskal(graph, explain):
    edges = []
    for u in graph:
        for v, weight in graph[u].items():
            edges.append((weight, u, v))

    edges = sorted(edges, key=lambda item: item[0])
    parent = {}
    rank = {}
    for node in graph:
        parent[node] = node
        rank[node] = 0

    result = []
    e = 0
    i = 0
    explain('Starting Kruskal\'s Algorithm')

    while e < len(graph) - 1:
        w, u, v = edges[i]
        i += 1
        x = find(parent, u)
        y = find(parent, v)
        if x != y:
            e += 1
            result.append((u, v, w))
            union(parent, rank, x, y)
            explain(f'Adding edge {u}-{v} with cost {w}')

    return result

# Fibonacci Sequence
def fibonacci(n):
    fib_sequence = [0, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[i-1] + fib_sequence[i-2])
    return fib_sequence

# Function to create buttons in columns
def create_buttons(inner_frame):
    algorithms = [
        "Bubble Sort", "Insertion Sort", "Quick Sort", "Merge Sort",
        "KMP Search", "Dijkstra's Algorithm", "Prim's Algorithm", 
        "Kruskal's Algorithm", "Fibonacci"
    ]

    # Create buttons in multiple columns
    for i, algo in enumerate(algorithms):
        button = tk.Button(inner_frame, text=algo, command=lambda a=algo: execute_algorithm(a), bg='lightblue', font=('Helvetica', 12))
        button.grid(row=i % 6, column=i // 6, pady=5, padx=10)

# Execute the selected algorithm
def execute_algorithm(algorithm):
    plt.ioff()  # Turn off interactive mode
    explanation_text.delete(1.0, tk.END)  # Clear previous explanation
    explain = lambda text: explanation_text.insert(tk.END, text + '\n')  # Function to update explanation

    if algorithm == "Bubble Sort":
        arr = [random.randint(1, 100) for _ in range(10)]
        explain(f'Initial array: {arr}')
        draw(arr)  # Display the initial array
        bubble_sort(arr, draw, 0.5, explain)

    elif algorithm == "Insertion Sort":
        arr = [random.randint(1, 100) for _ in range(10)]
        explain(f'Initial array: {arr}')
        draw(arr)  # Display the initial array
        insertion_sort(arr, draw, 0.5, explain)

    elif algorithm == "Quick Sort":
        arr = [random.randint(1, 100) for _ in range(10)]
        explain(f'Initial array: {arr}')
        draw(arr)  # Display the initial array
        quick_sort(arr, 0, len(arr) - 1, draw, 0.5, explain)

    elif algorithm == "Merge Sort":
        arr = [random.randint(1, 100) for _ in range(10)]
        explain(f'Initial array: {arr}')
        draw(arr)  # Display the initial array
        merge_sort(arr, draw, 0.5, explain)

    elif algorithm == "KMP Search":
        text = "ababcababcabc"
        pattern = "abc"
        index = kmp_search(text, pattern, explain)

    elif algorithm == "Dijkstra's Algorithm":
        graph = {
            'A': {'B': 1, 'C': 4},
            'B': {'A': 1, 'C': 2, 'D': 5},
            'C': {'A': 4, 'B': 2, 'D': 1},
            'D': {'B': 5, 'C': 1}
        }
        distances = dijkstra(graph, 'A', explain)
        explain(f"Distances from A: {distances}")

    elif algorithm == "Prim's Algorithm":
        graph = {
            'A': {'B': 1, 'C': 4},
            'B': {'A': 1, 'C': 2, 'D': 5},
            'C': {'A': 4, 'B': 2, 'D': 1},
            'D': {'B': 5, 'C': 1}
        }
        mst = prim(graph, explain)
        explain(f"Minimum Spanning Tree: {mst}")

    elif algorithm == "Kruskal's Algorithm":
        graph = {
            'A': {'B': 1, 'C': 4},
            'B': {'A': 1, 'C': 2, 'D': 5},
            'C': {'A': 4, 'B': 2, 'D': 1},
            'D': {'B': 5, 'C': 1}
        }
        mst = kruskal(graph, explain)
        explain(f"Minimum Spanning Tree: {mst}")

    elif algorithm == "Fibonacci":
        n = 10  # Calculate first 10 Fibonacci numbers
        fib_sequence = fibonacci(n)
        plt.ion()
        draw(fib_sequence)
        plt.title("Fibonacci Sequence")
        plt.show(block=False)  # Non-blocking show for Matplotlib
        plt.ioff()  # Turn off interactive mode again

# GUI setup
root = tk.Tk()
root.title("Algorithm Visualizer")

# Create a frame for buttons on the left
button_frame = tk.Frame(root)
button_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Create a scrollable frame for buttons
scrollbar = tk.Scrollbar(button_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Create a canvas to hold the buttons
button_canvas = tk.Canvas(button_frame, yscrollcommand=scrollbar.set)
button_canvas.pack(side=tk.LEFT)

# Configure the scrollbar
scrollbar.config(command=button_canvas.yview)

# Create a frame for the buttons inside the canvas
button_inner_frame = tk.Frame(button_canvas)
button_canvas.create_window((0, 0), window=button_inner_frame, anchor='nw')

# Create buttons
create_buttons(button_inner_frame)

# Update the scroll region
button_inner_frame.update_idletasks()
button_canvas.config(scrollregion=button_canvas.bbox("all"))

# Frame for the explanation text on the right
explanation_text = scrolledtext.ScrolledText(root, width=60, height=20, font=('Helvetica', 10))
explanation_text.pack(side=tk.RIGHT, padx=10, pady=10)

# Override close window event
def on_closing():
    global gui_active
    gui_active = False  # Set GUI active flag to False
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)  # Handle the close button

# Run the GUI
root.mainloop()
