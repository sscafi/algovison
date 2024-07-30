import random
import time
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq

# Global variable to track if the GUI is active
gui_active = True

# Example sorting algorithms
def bubble_sort(arr, draw, pause):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                draw(arr)
                time.sleep(pause)
                if not gui_active:  # Check if GUI is active
                    return  # Exit if GUI is closed

def insertion_sort(arr, draw, pause):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >=0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        draw(arr)
        time.sleep(pause)
        if not gui_active:  # Check if GUI is active
            return  # Exit if GUI is closed

def quick_sort(arr, low, high, draw, pause):
    if low < high:
        pi = partition(arr, low, high, draw, pause)
        quick_sort(arr, low, pi-1, draw, pause)
        quick_sort(arr, pi+1, high, draw, pause)

def partition(arr, low, high, draw, pause):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            draw(arr)
            time.sleep(pause)
            if not gui_active:  # Check if GUI is active
                return  # Exit if GUI is closed
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    draw(arr)
    time.sleep(pause)
    return i + 1

def merge_sort(arr, draw, pause):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L, draw, pause)
        merge_sort(R, draw, pause)

        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            draw(arr)
            time.sleep(pause)
            if not gui_active:  # Check if GUI is active
                return  # Exit if GUI is closed
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
            draw(arr)
            time.sleep(pause)
            if not gui_active:  # Check if GUI is active
                return  # Exit if GUI is closed

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
            draw(arr)
            time.sleep(pause)
            if not gui_active:  # Check if GUI is active
                return  # Exit if GUI is closed

# KMP Search Algorithm
def kmp_search(text, pattern):
    m = len(pattern)
    n = len(text)

    # Create lps array
    lps = [0] * m
    j = 0  # length of previous longest prefix suffix
    compute_lps_array(pattern, m, lps)

    i = 0  # index for text
    while n - i >= m:
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == m:
            return i - j  # Found the pattern
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

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
def dijkstra(graph, start):
    queue = []
    heapq.heappush(queue, (0, start))
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))

    return distances

# Prim's Algorithm
def prim(graph):
    start = next(iter(graph))
    visited = {start}
    edges = [(cost, start, to) for to, cost in graph[start].items()]
    heapq.heapify(edges)
    mst = []

    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst.append((frm, to, cost))
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

def kruskal(graph):
    result = []
    i = 0
    e = 0
    edges = []
    
    for u in graph:
        for v, w in graph[u].items():
            edges.append((w, u, v))

    edges.sort()
    parent = {}
    rank = {}

    for node in graph:
        parent[node] = node
        rank[node] = 0

    while e < len(graph) - 1:
        w, u, v = edges[i]
        i += 1
        x = find(parent, u)
        y = find(parent, v)

        if x != y:
            e += 1
            result.append((u, v, w))
            union(parent, rank, x, y)

    return result

# Dynamic Programming Example: Fibonacci
def fibonacci(n):
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

# Execute Algorithm
def execute_algorithm(algorithm):
    if gui_active:  # Check if the GUI is active
        if algorithm in ["Bubble Sort", "Insertion Sort", "Quick Sort", "Merge Sort"]:
            array = [random.randint(1, 100) for _ in range(20)]
            plt.ion()
            draw(array)

            if algorithm == "Bubble Sort":
                bubble_sort(array, draw, 0.1)
            elif algorithm == "Insertion Sort":
                insertion_sort(array, draw, 0.1)
            elif algorithm == "Quick Sort":
                quick_sort(array, 0, len(array) - 1, draw, 0.1)
            elif algorithm == "Merge Sort":
                merge_sort(array, draw, 0.1)

            plt.ioff()
            plt.show()  # Wait for user to close the plot

        elif algorithm == "KMP Search":
            text = "ababcababcabc"
            pattern = "abc"
            index = kmp_search(text, pattern)
            visualize(text, title=f'KMP Search - Searching for "{pattern}"')

        elif algorithm == "Dijkstra's Algorithm":
            graph = {
                'A': {'B': 1, 'C': 4},
                'B': {'A': 1, 'C': 2, 'D': 5},
                'C': {'A': 4, 'B': 2, 'D': 1},
                'D': {'B': 5, 'C': 1}
            }
            distances = dijkstra(graph, 'A')
            messagebox.showinfo("Dijkstra's Algorithm", f"Distances from A: {distances}")

        elif algorithm == "Prim's Algorithm":
            graph = {
                'A': {'B': 1, 'C': 4},
                'B': {'A': 1, 'C': 2, 'D': 5},
                'C': {'A': 4, 'B': 2, 'D': 1},
                'D': {'B': 5, 'C': 1}
            }
            mst = prim(graph)
            messagebox.showinfo("Prim's Algorithm", f"Minimum Spanning Tree: {mst}")

        elif algorithm == "Kruskal's Algorithm":
            graph = {
                'A': {'B': 1, 'C': 4},
                'B': {'A': 1, 'C': 2, 'D': 5},
                'C': {'A': 4, 'B': 2, 'D': 1},
                'D': {'B': 5, 'C': 1}
            }
            mst = kruskal(graph)
            messagebox.showinfo("Kruskal's Algorithm", f"Minimum Spanning Tree: {mst}")

        elif algorithm == "Fibonacci":
            n = 10  # Calculate first 10 Fibonacci numbers
            fib_sequence = fibonacci(n)
            plt.ion()
            draw(fib_sequence)
            plt.ioff()
            plt.show()  # Wait for user to close the plot

# GUI setup
root = tk.Tk()
root.title("Algorithm Visualizer")

def on_button_click(algorithm):
    global gui_active  # Allow access to the global variable
    if gui_active:  # Check if the GUI is active
        execute_algorithm(algorithm)

# Create buttons for algorithms
algorithms = [
    "Bubble Sort", "Insertion Sort", "Quick Sort", "Merge Sort",
    "KMP Search", "Dijkstra's Algorithm", "Prim's Algorithm", "Kruskal's Algorithm", "Fibonacci"
]
for algo in algorithms:
    button = tk.Button(root, text=algo, command=lambda a=algo: on_button_click(a), bg='lightblue', font=('Helvetica', 12))
    button.pack(pady=5, padx=10)

# Override close window event
def on_closing():
    global gui_active
    gui_active = False  # Set GUI active flag to False
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)  # Handle the close button

# Run the GUI
root.mainloop()
