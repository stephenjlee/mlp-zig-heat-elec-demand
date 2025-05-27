import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the filtered ResStock dataset
df = pd.read_csv("filtered_resstock_data.csv")

# Use reported total natural gas input from the meter
df["Total NG Input (MBtu)"] = df["Fuel Use: Natural Gas: Total (MBtu)"]

# Calculate total delivered thermal load (heating + hot water)
df["Total Delivered Load (MBtu)"] = (
    df["Load: Heating: Delivered (MBtu)"] +
    df["Load: Hot Water: Delivered (MBtu)"]
)

# Remove rows where total gas input is zero or missing
df = df[df["Total NG Input (MBtu)"] > 0]

# Constrained linear regression through the origin: y = mx
x = df["Total NG Input (MBtu)"]
y = df["Total Delivered Load (MBtu)"]
slope = np.sum(x * y) / np.sum(x ** 2)  # Least squares slope with intercept = 0

# Compute R^2 manually for fit through the origin
y_pred = slope * x
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Plot the data
plt.figure(figsize=(8, 5))  # Slightly smaller figure with same aspect ratio
plt.scatter(x, y, alpha=0.5, label="Individual Residences")

# Plot the origin-constrained trendline
x_line = np.linspace(x.min(), x.max(), 100)
y_line = slope * x_line
plt.plot(
    x_line,
    y_line,
    color='red',
    label=f"Fit: y = {slope:.4f}x  (R² = {r_squared:.2f})"
)

# Set axis labels and title
plt.xlabel("Total Natural Gas Input (MBtu)\n(Metered Fuel Use)", fontsize=11)
plt.ylabel("Total Delivered Thermal Load (MBtu)\n(Space Heating + Hot Water)", fontsize=11)
plt.title("Relationship Between Metered Natural Gas Input and Delivered Load", fontsize=13)

# Customize appearance
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
# plt.show()
plt.savefig('gas_input_vs_thermal_output.png')