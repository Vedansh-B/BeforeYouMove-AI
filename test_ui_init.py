import tkinter as tk
from chess_ui import ChessWorkbenchApp

# Test minimal UI initialization (without displaying)
print("Testing ChessWorkbenchApp initialization...")
try:
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    app = ChessWorkbenchApp(root)
    
    print("✓ ChessWorkbenchApp initialized successfully")
    print(f"✓ Board canvas created: {app.canvas is not None}")
    print(f"✓ Planning listbox created: {app.plans_listbox is not None}")
    print(f"✓ Depth spinbox value: {app.depth_spinbox.get()}")
    
    root.destroy()
    print("✓ UI test passed!")
except Exception as e:
    print(f"✗ UI test failed: {e}")
    import traceback
    traceback.print_exc()
