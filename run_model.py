import os, json, pickle
import pandas as pd
import numpy as np
import tensorflow as tf

keras = tf.keras

from ldf.models.model_ldf_scalar import LdfScalarModel


def run_ldf(args, x, application='heat'):
    # default config
    args['model_name'] = 'XLARGE_UNRESTRICTED'

    if (application == 'heat') or (application == 'ng'):
        args['load_json'] = os.path.join('model_data', 'heat-MLP-ZIG',
                                         'model_opt.json')
        args['load_weights'] = os.path.join('model_data', 'heat-MLP-ZIG',
                                            'model_opt.h5')
        args['output_path'] = os.path.join('out', 'heat-zig-out')
    elif application == 'elec':
        args['load_json'] = os.path.join('model_data', 'elec-MLP-ZIG',
                                         'model_opt.json')
        args['load_weights'] = os.path.join('model_data', 'elec-MLP-ZIG',
                                            'model_opt.h5')
        args['output_path'] = os.path.join('out', 'elec-zig-out')
    else:
        raise ValueError('Unknown application: {}'.format(application))

    # load data definitions
    x_scaler = pickle.load(
        open(os.path.join('model_data', 'heat_data', 'x_scaler.pkl'), 'rb'))
    with open(os.path.join('model_data', 'heat_data', 'data_keys.json'), 'r') as f:
        feats = json.load(f)

    # transform data
    x = x_scaler.transform(pd.DataFrame(x)[feats])

    # setup ldf model
    ldf_model = LdfScalarModel(
        verbose=1,
        gpu=0,
        error_metric='zig',
        model_name=args['model_name'],
        batch_size=1,
        output_dir=None,
        load_json=args['load_json'],
        load_weights=args['load_weights']
    )
    ldf_model.setup_model(X=x, load_model=True)

    # predict
    yhat = ldf_model.model_.predict(x.astype(np.float64), verbose=0)
    mean_preds, \
        std_preds, \
        preds_params, \
        preds_params_flat = \
        ldf_model.distn.interpret_predict_output(yhat)

    scale_factor = 1.0
    if application == 'heat':
        scale_factor = args['ng_to_heat_eff_factor'] * args['therms_to_kbtu_factor']

    # format output
    output = {}
    output['p_zero_before_scaling'] = preds_params['p_zero']
    output['shape_before_scaling'] = preds_params['shape']
    output['scale_before_scaling'] = preds_params['scale']
    output['mean'] = mean_preds * scale_factor

    if application == 'heat':
        output['unit'] = 'kBTU'
    if application == 'ng':
        output['unit'] = 'therms'
    if application == 'elec':
        output['unit'] = 'kWh'

    output['y_hat_90th_percentile'] = ldf_model.distn.ppf_params(
        np.repeat(0.9, preds_params['p_zero'].size), preds_params) * scale_factor
    output['y_hat_80th_percentile'] = ldf_model.distn.ppf_params(
        np.repeat(0.8, preds_params['p_zero'].size), preds_params) * scale_factor
    output['y_hat_70th_percentile'] = ldf_model.distn.ppf_params(
        np.repeat(0.7, preds_params['p_zero'].size), preds_params) * scale_factor
    output['y_hat_60th_percentile'] = ldf_model.distn.ppf_params(
        np.repeat(0.6, preds_params['p_zero'].size), preds_params) * scale_factor
    output['y_hat_50th_percentile'] = ldf_model.distn.ppf_params(
        np.repeat(0.5, preds_params['p_zero'].size), preds_params) * scale_factor
    output['y_hat_40th_percentile'] = ldf_model.distn.ppf_params(
        np.repeat(0.4, preds_params['p_zero'].size), preds_params) * scale_factor
    output['y_hat_30th_percentile'] = ldf_model.distn.ppf_params(
        np.repeat(0.3, preds_params['p_zero'].size), preds_params) * scale_factor
    output['y_hat_20th_percentile'] = ldf_model.distn.ppf_params(
        np.repeat(0.2, preds_params['p_zero'].size), preds_params) * scale_factor
    output['y_hat_10th_percentile'] = ldf_model.distn.ppf_params(
        np.repeat(0.1, preds_params['p_zero'].size), preds_params) * scale_factor

    return output


def main(args):
    args = {
        'ng_to_heat_eff_factor': 0.7512,
        'therms_to_kbtu_factor': 100.
    }

    # mock data inputs
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

    heat_output = run_ldf(args, x, application='heat')
    ng_output = run_ldf(args, x, application='ng')
    elec_output = run_ldf(args, x, application='elec')

    print(f"heating demand: {heat_output['mean']} {heat_output['unit']}")
    print(f"ng demand: {ng_output['mean']} {ng_output['unit']}")
    print(f"electricity demand: {elec_output['mean']} {elec_output['unit']}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
