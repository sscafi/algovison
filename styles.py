def configure_styles():
    import tkinter.ttk as ttk
    style = ttk.Style()
    
    # Configure the main style
    style.configure('.', font=('Helvetica', 10))
    
    # Configure specific widget styles
    style.configure('TFrame', background='#f0f0f0')
    style.configure('TLabel', background='#f0f0f0')
    style.configure('TButton', padding=5)
    style.configure('TEntry', padding=5)
    style.configure('TCombobox', padding=5)
    
    # Configure notebook style
    style.configure('TNotebook', background='#f0f0f0')
    style.configure('TNotebook.Tab', padding=[10, 5])
    
    # Configure label frame style
    style.configure('TLabelframe', background='#f0f0f0')
    style.configure('TLabelframe.Label', background='#f0f0f0')
