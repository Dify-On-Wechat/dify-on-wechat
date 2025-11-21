import os
import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd


def process_excel(file_path: str, column: str) -> str:
    """Load an Excel file and remove rows where given column is not numeric."""
    df = pd.read_excel(file_path)

    # Allow column name or 1-based column index
    if column.isdigit():
        idx = int(column) - 1
        if idx < 0 or idx >= len(df.columns):
            raise ValueError("Column index out of range")
        col = df.columns[idx]
    else:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found")
        col = column

    numeric = pd.to_numeric(df[col], errors="coerce")
    cleaned = df[numeric.notnull()]

    base, ext = os.path.splitext(file_path)
    out_file = f"{base}_cleaned{ext}"
    cleaned.to_excel(out_file, index=False)
    return out_file


class ExcelCleaner(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Excel Cleaner")
        self.geometry("300x150")
        self.file_path = ""

        self.btn_load = tk.Button(self, text="Load Excel", command=self.load_file)
        self.btn_load.pack(pady=10)

        self.column_label = tk.Label(self, text="Column (name or index):")
        self.column_label.pack()
        self.column_entry = tk.Entry(self)
        self.column_entry.pack(pady=5)

        self.btn_process = tk.Button(self, text="Process", command=self.process)
        self.btn_process.pack(pady=10)

    def load_file(self):
        path = filedialog.askopenfilename(title="Select Excel file", filetypes=[("Excel files", "*.xlsx *.xls")])
        if path:
            self.file_path = path
            messagebox.showinfo("File Selected", path)

    def process(self):
        if not self.file_path:
            messagebox.showwarning("No File", "Please load an Excel file first")
            return
        column = self.column_entry.get().strip()
        if not column:
            messagebox.showwarning("No Column", "Please specify a column")
            return
        try:
            out = process_excel(self.file_path, column)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        else:
            messagebox.showinfo("Success", f"Processed file saved as:\n{out}")


if __name__ == "__main__":
    app = ExcelCleaner()
    app.mainloop()
