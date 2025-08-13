import tkinter as tk
from tkinter import ttk, scrolledtext

class ExplanationWindow:
    def __init__(self, parent):
        self.parent = parent
        self.frame = ttk.LabelFrame(parent, text="Algorithm Explanation")
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.text = scrolledtext.ScrolledText(
            self.frame, wrap=tk.WORD, width=40, height=10
        )
        self.text.pack(fill=tk.BOTH, expand=True)
        
        # Add clear button
        ttk.Button(self.frame, text="Clear", command=self.clear).pack()
    
    def add_message(self, message):
        self.text.insert(tk.END, message + "\n")
        self.text.see(tk.END)
    
    def clear(self):
        self.text.delete(1.0, tk.END)
