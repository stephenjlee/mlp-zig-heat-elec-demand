import os
import csv

# Updated base directory
base_dir = "<add path to your base directory here>"

# Categories to check for zero values
zero_check_categories = [
    "End Use: Electricity: Heating (MBtu)",
    "End Use: Electricity: Hot Water (MBtu)",
    "Load: Hot Water: Solar Thermal (MBtu)",
    "End Use: Fuel Oil: Heating (MBtu)",
    "End Use: Propane: Heating (MBtu)",
    "End Use: Wood Cord: Heating (MBtu)",
    "End Use: Wood Pellets: Heating (MBtu)",
    "End Use: Coal: Heating (MBtu)"
]

# Categories to extract if the above are zero
output_categories = [
    "Fuel Use: Natural Gas: Total (MBtu)",
    "End Use: Natural Gas: Heating (MBtu)",
    "End Use: Natural Gas: Hot Water (MBtu)",
    "End Use: Natural Gas: Range/Oven (MBtu)",
    "Load: Heating: Delivered (MBtu)",
    "Load: Hot Water: Delivered (MBtu)",
    "Load: Hot Water: Tank Losses (MBtu)"
]

# Path to the output CSV
output_csv = "filtered_resstock_data.csv"

# Store rows to write
output_data = []

# Traverse each 'run#' folder
for run_folder in os.listdir(base_dir):
    full_run_path = os.path.join(base_dir, run_folder)

    if os.path.isdir(full_run_path) and run_folder.startswith("run"):
        run_id = run_folder.replace("run", "")  # just the number part
        results_path = os.path.join(full_run_path, "run", "results_annual.csv")

        if not os.path.exists(results_path):
            continue  # skip if file doesn't exist

        # Read the CSV file into a dictionary
        data = {}
        try:
            with open(results_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) < 2:
                        continue
                    category, value = row[0].strip(), row[1].strip()
                    try:
                        data[category] = float(value) if value else 0.0
                    except ValueError:
                        data[category] = 0.0  # fallback in case of bad value
        except Exception as e:
            print(f"Error reading {results_path}: {e}")
            continue

        # Check if zero for all target categories
        if all(data.get(category, 0.0) == 0.0 for category in zero_check_categories):
            row_data = [run_id]
            for category in output_categories:
                row_data.append(data.get(category, 0.0))
            output_data.append(row_data)

# Write everything to a new CSV
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Run ID"] + output_categories)
    writer.writerows(output_data)
    
print(f"Found {len(output_data)} runs matching the filter criteria.")
print(f"Filtered data saved to: {output_csv}")