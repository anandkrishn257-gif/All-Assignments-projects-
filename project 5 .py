import tkinter as tk
from sklearn.linear_model import LinearRegression

# ---------------- Dataset ----------------
PM25 = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
PM10 = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
NO2  = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
CO   = [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8]

AQI  = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130]



# Create input data
X = list(zip(PM25, PM10, NO2, CO))

# ---------------- Train Model ----------------
model = LinearRegression()
model.fit(X, AQI)

# ---------------- Prediction Function ----------------
def predict_aqi():
    pm25 = float(entry_pm25.get())
    pm10 = float(entry_pm10.get())
    no2 = float(entry_no2.get())
    co = float(entry_co.get())

    prediction = model.predict([[pm25, pm10, no2, co]])

    result.config(
        text=f"Predicted AQI: {prediction[0]:.2f}"
    )

# ---------------- Tkinter Window ----------------
window = tk.Tk()
window.title("Air Quality Prediction")
window.geometry("400x400")

title = tk.Label(
    window,
    text="Air Quality Prediction",
    font=("Arial", 20, "bold")
)
title.pack(pady=20)

# PM2.5
tk.Label(window, text="PM2.5:", font=("Arial", 12)).pack()
entry_pm25 = tk.Entry(window)
entry_pm25.pack(pady=5)

# PM10
tk.Label(window, text="PM10:", font=("Arial", 12)).pack()
entry_pm10 = tk.Entry(window)
entry_pm10.pack(pady=5)

# NO2
tk.Label(window, text="NO2:", font=("Arial", 12)).pack()
entry_no2 = tk.Entry(window)
entry_no2.pack(pady=5)

# CO
tk.Label(window, text="CO:", font=("Arial", 12)).pack()
entry_co = tk.Entry(window)
entry_co.pack(pady=5)

# Button
button = tk.Button(
    window,
    text="Predict AQI",
    command=predict_aqi,
    font=("Arial", 12, "bold")
)
button.pack(pady=20)

# Result
result = tk.Label(
    window,
    text="Predicted AQI: ",
    font=("Arial", 16, "bold")
)
result.pack(pady=10)

window.mainloop()
