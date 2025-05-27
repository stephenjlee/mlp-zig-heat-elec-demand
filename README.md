# mlp-zig-heat-elec-demand

Stephen J. Lee

Last updated: May 2025

Please reach out stephenl@umass.edu with any questions.

------------------

### License

This code and model is licensed under the MIT License. Please see the file named `LICENSE` in this directory for details.

### Attribution

If you use this model, please cite our corresponding paper:

  * Stephen J. Lee and Cailinn Drouin. Forecasting Residential Heating and Electricity Demand with Scalable, High-Resolution, Open-Source Models. arXiv, 2025.

  * Bibtex:
    ~~~
    @article{lee2025forecasting,
      title={Forecasting Residential Heating and Electricity Demand with Scalable, High-Resolution, Open-Source Models},
      author={Lee, Stephen J. and Drouin, Cailinn},
      journal={arXiv preprint},
      year={2025},
      note={Working Paper}
    }
    ~~~


### Overview

This repository provides trained Zero-Inflated Gamma (ZIG) machine learning models for forecasting building-level heating and electricity demand. The models enable users to run high-resolution demand estimations for arbitrary buildings using remote sensing and geospatial features as inputs. By offering an open-source, scalable framework, this repository supports power system planning and helps accelerate the decarbonization of the economy and energy sector.

All data in this repository has been fully anonymized, ensuring no personally identifiable information (PII) is included. The models included in this repository are compact—each less than 50 KB in size—while being trained on terabytes of consumption data. This highly compressed model effectively maps widely available remote sensing features to probabilistic heating and electricity demand outputs, facilitating demand estimation at the building level. It reflects probable demand based off of these publicly available remote sensing features and in no way provides actual consumption values for any given consumer. 

In fact, heating and electricity demand are inherently stochastic. They are shaped not only by building attributes and weather conditions but also by occupant behavior; households vary in their temperature preferences, heating schedules, and use of supplemental heating sources such as space heaters or fireplaces. Additionally, building insulation quality varies widely, with structures experiencing heterogeneous persistent heating leaks that can last for years or decades. Similarly, electricity consumption fluctuates based on factors such as appliance efficiency, the presence of high-power devices, and individual lifestyle patterns and preferences. These behavioral variations introduce inherent noise into the data and create a fundamental performance ceiling that all models must contend with.

The current models are specifically designed for the U.S. Mid-Atlantic region, where they have been validated with empirical data. Future research will explore the generalizability of this approach to other regions and assess its scalability for broader applications in energy system modeling and policy analysis.


### Environment setup

* Use Mamba to install the configuration sepecified in `envs/environment.yml`.
    ~~~~
    mamba env create -f environment.yml
    ~~~~

* Alternatively, use pip (python version 3.9 tested):
    ~~~~
    python -m venv .mlp-zig
    .mlp-zig\Scripts\activate
    pip install -r requirements.txt
    ~~~~

### Local environment variable setup

* Please create or edit a text file in your project root called *.env*, and set its contents to:
    ~~~~
    PROJECT_ROOT=<path to your project root>
    ~~~~

------------------

### Running model

* **Run** `run_model.py`.
    ~~~~
    python run_model.py
    ~~~~

* **Data inputs**: We have provided the following anonymized building-level default inputs. Please see our paper for our definition of these features and their corresponding data sources. You may subsitute data for arbitrary buildings in your dataset. You may also use the list structure to provide inputs for multiple buildings.
    ~~~~
        x = [{'area_in_meters': 100.,
          'bldgs_1000m_away_n_bldgs': 2000.,
          'bldgs_1000m_away_avg_area': 100.,
          'bldgs_1000m_away_std_area': 400.,
          'bldg_height_wsf3dv02': 100.,
          'num_floors': 0.0,
          'ntl2022_N1': 40.,
          'ntl2022_N11': 40.,
          'ntl2022_N51': 40.,
          'lulc2022_built_area_N1': 1.0,
          'lulc2022_built_area_N11': 1.0,
          'lulc2022_built_area_N51': 1.0,
          'lulc2022_crops_N1': 0.0,
          'lulc2022_crops_N11': 0.0,
          'lulc2022_crops_N51': 0.0,
          'lulc2022_trees_N1': 0.0,
          'lulc2022_trees_N11': 0.0,
          'lulc2022_trees_N51': 0.0,
          'lulc2022_water_N1': 0.0,
          'lulc2022_water_N11': 0.0,
          'lulc2022_water_N51': 0.0,
          'ookla_fixed_20220101_avg_d_kbps': 500000.0,
          'ookla_fixed_20220101_avg_lat_ms': 50.0,
          'ookla_fixed_20220101_avg_u_kbps': 50000.0,
          'ookla_fixed_20220101_devices': 5.0,
          'ookla_fixed_20220101_tests': 5.0,
          'ookla_mobile_20220101_avg_d_kbps': 0.0,
          'ookla_mobile_20220101_avg_lat_ms': 0.0,
          'ookla_mobile_20220101_avg_u_kbps': 0.0,
          'ookla_mobile_20220101_devices': 0.0,
          'ookla_mobile_20220101_tests': 0.0,
          'total_precipitation': 0.0,
          'surface_pressure': 100000.,
          'soil_temperature_level_1': 270.,
          'instantaneous_10m_wind_gust': 10.,
          '2m_temperature': 270.,
          'snow_depth': 0.0,
          'surface_net_thermal_radiation': -365000.,
          '10m_u_component_of_wind': 5.,
          'total_cloud_cover': 0.0,
          'high_vegetation_cover': 0.5,
          '10m_v_component_of_wind': 5.,
          'surface_net_solar_radiation': 3.e-12,
          'total_precipitation_m1': 0.0,
          'surface_pressure_m1': 101500.,
          'soil_temperature_level_1_m1': 271.,
          'instantaneous_10m_wind_gust_m1': 5.,
          '2m_temperature_m1': 268.,
          'snow_depth_m1': 0.0,
          'surface_net_thermal_radiation_m1': -360000.,
          '10m_u_component_of_wind_m1': 3.,
          'total_cloud_cover_m1': 0.0,
          'high_vegetation_cover_m1': 0.5,
          '10m_v_component_of_wind_m1': -2.,
          'surface_net_solar_radiation_m1': 3.e-12,
          'total_precipitation_m2': 0.0,
          'surface_pressure_m2': 101500.,
          'soil_temperature_level_1_m2': 272.,
          'instantaneous_10m_wind_gust_m2': 5.,
          '2m_temperature_m2': 268.,
          'snow_depth_m2': 0.0,
          'surface_net_thermal_radiation_m2': -364000.,
          '10m_u_component_of_wind_m2': 3.,
          'total_cloud_cover_m2': 0.0,
          'high_vegetation_cover_m2': 0.5,
          '10m_v_component_of_wind_m2': -2.,
          'surface_net_solar_radiation_m2': 3.e-12,
          'Monday': True,
          'Tuesday': False,
          'Wednesday': False,
          'Thursday': False,
          'Friday': False,
          'Saturday': False,
          'Sunday': False,
          'hour_of_day': 5.0}]
    ~~~~
### Output format

The model outputs probabilistic estimates of consumption for **heating**, **natural gas**, and **electricity** demand. These outputs are structured according to a zero-inflated gamma (ZIG) distribution and differ by application in terms of units and scaling:

#### General Output Parameters

The following values are returned for each building-level prediction:

  - **p_zero_before_scaling**: ZIG parameter: Probability of zero consumption.
  - **shape_before_scaling**: ZIG shape parameter.
  - **scale_before_scaling**: ZIG scale parameter.
  - **mean**: Expected mean of the predicted values.
  - **std**: Standard deviation of the predicted values.
  - **y_hat_90th_percentile**: 90th percentile estimate of predicted values.
  - **y_hat_80th_percentile**: 80th percentile estimate of predicted values.
  - **y_hat_70th_percentile**: 70th percentile estimate of predicted values.
  - **y_hat_60th_percentile**: 60th percentile estimate of predicted values.
  - **y_hat_50th_percentile**: 50th percentile estimate (median) of predicted values.
  - **y_hat_40th_percentile**: 40th percentile estimate of predicted values.
  - **y_hat_30th_percentile**: 30th percentile estimate of predicted values.
  - **y_hat_20th_percentile**: 20th percentile estimate of predicted values.
  - **y_hat_10th_percentile**: 10th percentile estimate of predicted values.

Each percentile reflects a specific quantile of the estimated demand distribution, scaled appropriately for the application.

#### Units and Scaling Logic

The model supports three types of demand estimation, with output formatted as follows:

- **Heating demand (`application='heat'`)**:
  - **Unit**: `kBTU`
  - **Scaling**: Applies a conversion factor equal to  
    `ng_to_heat_eff_factor * therms_to_kbtu_factor`
    - Default values:
      - `ng_to_heat_eff_factor` = `0.7512` (based on ResStock simulations)
      - `therms_to_kbtu_factor` = `100.0`
  - This reflects the conversion from natural gas consumption to useful delivered heat.

- **Natural Gas demand (`application='ng'`)**:
  - **Unit**: `therms`
  - **Scaling**: No scaling applied. Outputs reflect raw model predictions.

- **Electricity demand (`application='elec'`)**:
  - **Unit**: `kWh`
  - **Scaling**: No scaling applied. Outputs reflect raw model predictions.

The scaling for heating demand ensures that results reflect delivered heat (for space heating and hot water) rather than consumed fuel (e.g. including losses and fuel for cooking or miscellaneous other applications), enabling comparison across fuel types and building energy performance.

#### Unit Field

Each output dictionary includes a `"unit"` field that explicitly indicates whether the output values are in `kBTU`, `therms`, or `kWh`, depending on the application.

------------------

### Determining natural gas to heating demand efficiency factor

We employ ResStock simulations to compute a simple efficiency factor (η) for approximately converting between natural gas and heating demand (for space heating and hot water). Please see the readme file and repo contents within our folder `resstock_efficiency_factor_analysis` for more details. 
