import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# -------------------- UI COLORS & FONTS --------------------
BG_COLOR = "#FFA500"   # Light Orange
BTN_COLOR = "#FF8C00"
TEXT_COLOR = "#1f1f1f"

TITLE_FONT = ("Georgia", 18, "bold")
LABEL_FONT = ("Trebuchet MS", 11)
BTN_FONT = ("Trebuchet MS", 11, "bold")
OUTPUT_FONT = ("Georgia", 10)

# -------------------- GLOBAL DATA --------------------
df = None

# -------------------- FUNCTIONS --------------------
def load_csv():
    global df
    try:
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, "CSV loaded successfully.\n")
        output_text.insert(tk.END, f"Rows: {len(df)}\nColumns: {list(df.columns)}\n")

    except Exception as e:
        messagebox.showerror("Error", str(e))


def run_model():
    global df
    if df is None:
        messagebox.showwarning("No Data", "Please load a CSV file first.")
        return

    try:
        # Feature Engineering
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofweek'] = df['date'].dt.dayofweek

        X = df[['day', 'month', 'year', 'dayofweek']]
        y = df['sales']

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, shuffle=False, test_size=0.2
        )

        # Forecast Model
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mape = mean_absolute_percentage_error(y_test, predictions)

        # Anomaly Detection
        iso = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly'] = iso.fit_predict(df[['sales']])
        anomalies = df[df['anomaly'] == -1]

        # Output
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, "MODEL RESULTS\n")
        output_text.insert(tk.END, "-" * 40 + "\n")
        output_text.insert(tk.END, f"RMSE: {rmse:.2f}\n")
        output_text.insert(tk.END, f"MAPE: {mape:.2%}\n")
        output_text.insert(tk.END, f"Anomalies Detected: {len(anomalies)}\n\n")

        output_text.insert(tk.END, "Anomaly Dates & Sales:\n")
        for _, row in anomalies.iterrows():
            output_text.insert(
                tk.END, f"{row['date'].date()}  →  {row['sales']}\n"
            )

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df['date'], df['sales'], label="Sales")
        plt.scatter(anomalies['date'], anomalies['sales'], label="Anomaly")
        plt.title("Sales Forecast & Anomaly Detection")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", str(e))


# -------------------- UI SETUP --------------------
root = tk.Tk()
root.title("Hybrid Sales Forecast & Anomaly Detection")
root.geometry("820x520")
root.configure(bg=BG_COLOR)

# Title
title_label = tk.Label(
    root,
    text="Hybrid Predictive System",
    bg=BG_COLOR,
    fg=TEXT_COLOR,
    font=TITLE_FONT
)
title_label.pack(pady=10)

subtitle_label = tk.Label(
    root,
    text="Sales Forecasting + Anomaly Detection",
    bg=BG_COLOR,
    fg=TEXT_COLOR,
    font=LABEL_FONT
)
subtitle_label.pack()

# Buttons Frame
btn_frame = tk.Frame(root, bg=BG_COLOR)
btn_frame.pack(pady=15)

load_btn = tk.Button(
    btn_frame,
    text="Load Sales CSV",
    command=load_csv,
    bg=BTN_COLOR,
    fg="white",
    font=BTN_FONT,
    width=18
)
load_btn.grid(row=0, column=0, padx=10)

run_btn = tk.Button(
    btn_frame,
    text="Run Prediction",
    command=run_model,
    bg=BTN_COLOR,
    fg="white",
    font=BTN_FONT,
    width=18
)
run_btn.grid(row=0, column=1, padx=10)

# Output Box
output_frame = tk.Frame(root, bg=BG_COLOR)
output_frame.pack(fill="both", expand=True, padx=15, pady=10)

output_text = tk.Text(
    output_frame,
    font=OUTPUT_FONT,
    wrap="word"
)
output_text.pack(fill="both", expand=True)

# Footer
footer = tk.Label(
    root,
    text="Single-File Python | Tkinter UI | ML Based System",
    bg=BG_COLOR,
    fg=TEXT_COLOR,
    font=("Trebuchet MS", 9)
)
footer.pack(pady=5)

root.mainloop()
