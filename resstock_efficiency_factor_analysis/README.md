
# ResStock Gas-to-Heat Efficiency Analysis

This section of our repository provides an analysis of the relationship between metered natural gas usage and delivered thermal energy in residential buildings. Using simulated data derived from the [NREL ResStock](https://www.nrel.gov/buildings/resstock) v3.3.0 model, it estimates the efficiency of gas-based heating and hot water systems.

The core objective is to compute an empirical **efficiency factor** (η) using the form:

```
Delivered Load = η × Fuel Use
```

This factor can inform building decarbonization strategies, fuel-switching impact models, and energy retrofit planning.

---

## Contents

### Data File

- **`filtered_resstock_data.csv`**  
  A curated subset of ResStock v3.3.0 simulation outputs, filtered to include only households that rely on **natural gas for both space heating and water heating**.  
  The data includes:
  - `Fuel Use: Natural Gas: Total (MBtu)` – total gas use measured at the meter
  - End-use categories:
    - `End Use: Natural Gas: Heating (MBtu)`
    - `End Use: Natural Gas: Hot Water (MBtu)`
    - `End Use: Natural Gas: Range/Oven (MBtu)`
  - Delivered energy:
    - `Load: Heating: Delivered (MBtu)`
    - `Load: Hot Water: Delivered (MBtu)`
    - `Load: Hot Water: Tank Losses (MBtu)`

---

### Filtering Logic

- **`resstock_filter.py`**  
  A script to process raw ResStock output folders and create `filtered_resstock_data.csv`.  
  Filtering logic:
  - **Excludes** any household using **electricity, fuel oil, propane, wood, coal, or solar thermal** for heating or hot water.
  - **Includes** only homes with nonzero values in at least one of:
    - Gas space heating
    - Gas hot water
    - Gas cooking
  - Extracts both end-use and delivered load metrics for analysis.

---

### Efficiency Analysis

- **`plot_gas_input_vs_thermal_output.py`**  
  This script generates a scatter plot comparing:
  - **X-axis**: `Fuel Use: Natural Gas: Total (MBtu)` (meter-level input)
  - **Y-axis**: `Load: Heating + Hot Water Delivered (MBtu)` (heating output)  
  A linear regression through the origin (`y = ηx`) is fit to the data:
  - The slope η represents the empirical **conversion efficiency** from natural gas input to delivered heat.
  - The resulting R² value quantifies the linearity of the relationship. 
  - This approach assumes **zero intercept** for simplication, allowing us to employ a simple learned efficiency factor when converting from fuel use to heating in varied contexts.
  - Our logic is that the `Fuel Use: Natural Gas: Total` field includes all natural gas consumed by the home, not just what’s used for heating and hot water. By using the **efficiency factor η**, we remove gas fuel use for non-heating sources (e.g. cooking range/oven, and marginal use from potential others: dryers, pilot lights, burner standby losses, fireplaces, pools). 

---

## How to Use

1. **Prepare the dataset (optional)**  
   If needed, re-run the data filter:
   ```bash
   python resstock_filter.py
   ```

2. **Run the analysis**  
   Generate the efficiency plot:
   ```bash
   python plot_gas_input_vs_thermal_output.py
   ```


---

## Acknowledgments

Data and simulation engine:  
- [NREL ResStock](https://www.nrel.gov/buildings/resstock) v3.3.0  
This analysis was performed independently using publicly available simulation outputs.