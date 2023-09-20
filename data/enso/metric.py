''' Collection of metrics.

@Author  :   Jakob Schlör 
@Time    :   2022/10/18 15:29:31
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as stats
from scipy.fft import fft, fftfreq

from stfor.utils import preproc

def power_spectrum(data):
    """Compute power spectrum.

    Args:
        data (np.ndarray): Data of dimension (n_feat, n_time) 

    Returns:
        xf (np.ndarray): Frequencies of dimension (n_time//2) 
        yf (np.ndarray): Power spectrum of dimension (n_feat, n_time//2) 
    """
    n_feat, n_time = data.shape
    yf = []
    for i in range(n_feat):
        yf.append(fft(data[i,:]))

    xf = fftfreq(n_time, 1)[:n_time//2]
    yf = 2./n_time * np.abs(yf)[:, :n_time//2]
    return xf, yf 


def anomaly_correlation_coefficient(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Anomaly correlation coefficient.

    Args:
        x (np.ndarray): Ground truth data of size (n_time, n_features)
        x_hat (np.ndarray): Prediction of size (n_time, n_features)

    Returns:
        acc (np.ndarray): ACC of size (n_features) 
    """
    acc = np.array(
        [stats.pearsonr(x[:, i], x_hat[:, i])[0]
         for i in range(x.shape[1])]
    )
    return acc


def pattern_correlation(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Pattern correlation coefficient.

    Args:
        x (np.ndarray): Ground truth data of size (n_time, n_features)
        x_hat (np.ndarray): Prediction of size (n_time, n_features)

    Returns:
        acc (np.ndarray): Pattern correlation of size (n_times) 
    """
    pattern_corr = np.array(
        [stats.pearsonr(x[i, :], x_hat[i, :])[0]
         for i in range(x.shape[0])]
    )
    return pattern_corr


def frcst_metrics(target: xr.Dataset, frcst: xr.Dataset) -> dict:
    """Metrics for forecast in data space.

    Args:
        target (xr.Dataset): Target data of dimension (time, lat, lon).
        frcst (xr.Dataset): Forecast of dimension (time, lat, lon).

    Returns:
        dict: Metrics include: 
            - 'acc' of shape (lat, lon),
            - 'pattern_corr' of shape (time)
            - 'se' of shape (time, lat, lon)
            - 'mse' of shape (lat, lon)
        
    """
    # Correlation metrics
    pattern_corr = []
    acc = []
    for var in list(target.data_vars):
        temp_x, idx_nNaN = preproc.map2flatten(target[var])
        temp_x_frcst, _ = preproc.map2flatten(frcst[var])
        # ACC
        print("Compute ACC!")
        temp_acc = anomaly_correlation_coefficient(temp_x.data, temp_x_frcst.data)
        acc.append(preproc.flattened2map(temp_acc, idx_nNaN))

        # Pattern correlation
        print("Compute Pattern Correlation!")
        pattern_corr_temp = pattern_correlation(temp_x.data, temp_x_frcst.data)
        pattern_corr.append(xr.DataArray(
            data=pattern_corr_temp, coords={'time': temp_x['time']}, name=var
        ))

    acc = xr.merge(acc)
    pattern_corr = xr.merge(pattern_corr)

    # MSE
    print("Compute SE!")
    se = (target - frcst)**2
    mse = se.mean(dim='time', skipna=True)
    skill = 1 - np.sqrt(mse)/target.std(dim='time', skipna=True)
    mse_month = se.groupby('time.month').mean(dim='time', skipna=True)
    skill_month = 1 - np.sqrt(mse_month) / target.groupby('time.month').std(dim='time', skipna=True)

    # Store metrics
    return {'acc': acc, 'pattern_corr': pattern_corr, 'se': se, 'mse': mse,
            'mse_month': mse_month, 'skill': skill, 'skill_month': skill_month}


def frcst_metrics_per_month(target: xr.Dataset, frcst: xr.Dataset) -> dict:
    """Metrics for forecast in data space for each month seperately.

    Args:
        target (xr.Dataset): Target data of dimension (time, lat, lon).
        frcst (xr.Dataset): Forecast of dimension (time, lat, lon).

    Returns:
        dict: Metrics include: 
            - 'acc' of shape (lat, lon, month),
            - 'pattern_corr' of shape (month)
            - 'mse' of shape (lat, lon, month)
        
    """
    # Correlation metrics
    pattern_corr = []
    acc = []
    for var in list(target.data_vars):
        temp_x, idx_nNaN = preproc.map2flatten(target[var])
        temp_x_frcst, _ = preproc.map2flatten(frcst[var])

        # ACC
        print("Compute monthly ACC!")
        acc_monthly_var = []
        for m in np.unique(temp_x.time.dt.month):
            temp_acc = anomaly_correlation_coefficient(
                temp_x.isel(time=np.where(temp_x.time.dt.month == m)[0]),
                temp_x_frcst.isel(time=np.where(temp_x_frcst.time.dt.month == m)[0])
            )
            acc_monthly_var.append(preproc.flattened2map(temp_acc, idx_nNaN))
        acc.append(
            xr.concat(acc_monthly_var, dim=pd.Index(np.unique(temp_x.time.dt.month), name='month'))
        )

        # Pattern correlation
        print("Compute monthly Pattern Correlation!")
        pattern_corr_temp = xr.DataArray(
            data=pattern_correlation(temp_x.data, temp_x_frcst.data),
            coords={'time': temp_x['time']}, name=var
        )
        pattern_corr.append(
            pattern_corr_temp.groupby('time.month').mean(dim='time', skipna=True)
        )

    acc = xr.merge(acc)
    pattern_corr = xr.merge(pattern_corr)

    # MSE
    print("Compute monthly MSE!")
    se = (target - frcst)**2
    mse = se.groupby('time.month').mean(dim='time', skipna=True)

    # Store metrics
    return {'acc': acc, 'pattern_corr': pattern_corr, 'mse': mse}


def random_monthly_samples(ds: xr.Dataset, n_samples: int = 200) -> xr.Dataset:
    """Randomly samples the dataset by sampling months equally.

    Args:
        ds (xr.Dataset): Dataset to sample from 
        n_samples (int, optional): Number of samples. Defaults to 200.

    Returns:
        xr.Dataset: Subsampled dataset
    """
    # Get the unique months in the dataset
    months = np.unique(ds.time.dt.month)

    # Calculate the number of samples to draw per month
    n_per_month = int(np.ceil(n_samples / len(months)))

    # Initialize an empty list to hold the selected samples
    selected_samples = []

    # Loop over the months and select samples
    for month in months:
        # Get the indices of the time steps in the current month
        month_indices = np.where(ds.time.dt.month == month)[0]

        # Draw n_per_month random indices from the current month
        selected_indices = np.random.choice(month_indices, size=min(
            n_per_month, len(month_indices)), replace=True)

        # Add the selected samples to the list
        selected_samples.append(ds.isel(time=selected_indices))

    # Concatenate the selected samples into a new dataset
    selected_ds = xr.concat(selected_samples, dim='time')

    return selected_ds


def mean_diff(model1_skill: xr.DataArray, model2_skill: xr.DataArray,
              dim: str ='sample') -> list:
    """Compute difference of means and check for their statistical significance.
    
    We use the t-test for two sample groups and one sample groups.

    Args:
        model1_skill (xr.DataArray): Skill of model 1 with dimension 'ensemble'. 
        model2_skill (xr.DataArray): Skill of model 2 with dimension 'ensemble'.
        dim (str): Name of dimension the statistics is computed over

    Returns:
        diff (xr.DataArray): Difference of ensemble means.
        pvalues (xr.DataArray): Pvalues of ttest.
    """
    diff = model1_skill.mean(dim=dim, skipna=True) - model2_skill.mean(dim=dim, skipna=True)
    axis = int(np.where(np.array(model1_skill.dims) == dim)[0])
    if (len(model1_skill[dim]) > 1) and (len(model2_skill[dim]) > 1):
        statistic, pvalue = stats.ttest_ind(model1_skill.data, model2_skill.data, axis=axis, alternative='two-sided') 
    elif (len(model1_skill[dim]) > 1) and (len(model2_skill[dim]) == 1):
        statistic, pvalue = stats.ttest_1samp(model1_skill.data, model2_skill.mean(dim=dim), axis=axis, alternative='two-sided') 
    elif (len(model1_skill[dim]) == 1) and (len(model2_skill[dim]) > 1):
        statistic, pvalue = stats.ttest_1samp(model2_skill.data, model1_skill.mean(dim=dim), axis=axis, alternative='two-sided') 
    else:
        print(f"No samples given!")
        return diff, None

    pvalues = xr.DataArray(
        pvalue, coords=diff.coords,
    )

    return diff, pvalues


def listofdicts_to_dict(array_of_dicts: list) -> dict:
    """Convert list of dictionaries with same keys to dictionary of lists.

    Args:
        array_of_dicts (list): List of dictionaries with same keys. 

    Returns:
        dict: Dictionary of lists.
    """
    dict_of_arrays = {}
    for key in array_of_dicts[0].keys():
        dict_of_arrays[key] = [d[key] for d in array_of_dicts]
    return dict_of_arrays


def get_nino_indices(ssta, time_range=None, monthly=False, antimeridian=False):
    """Returns the time series of the Nino 1+2, 3, 3.4, 4, 5

    Args:
        ssta (xr.dataarray): Sea surface temperature anomalies.
        time_range (list, optional): Select only a certain time range.
        monthly (boolean): Averages time dimensions to monthly. 
                            Default to True.

    Returns:
        [type]: [description]
    """
    lon_range = [-90, -80] if antimeridian is False else preproc.get_antimeridian_coord([-90, -80])
    nino12, nino12_std = preproc.get_mean_time_series(
        ssta, lon_range=lon_range,
        lat_range=[-10, 0], time_roll=0
    )
    nino12.name = 'nino12'

    lon_range = [-150, -90] if antimeridian is False else preproc.get_antimeridian_coord([-150, -90])
    nino3, nino3_std = preproc.get_mean_time_series(
        ssta, lon_range=lon_range,
        lat_range=[-5, 5], time_roll=0
    )
    nino3.name = 'nino3'

    lon_range = [-170, -120] if antimeridian is False else preproc.get_antimeridian_coord([-170, -120])
    nino34, nino34_std = preproc.get_mean_time_series(
        ssta, lon_range=lon_range,
        lat_range=[-5, 5], time_roll=0
    )
    nino34.name = 'nino34'

    lon_range = [160, -150] if antimeridian is False else preproc.get_antimeridian_coord([160, -150])
    nino4, nino4_std = preproc.get_mean_time_series(
        ssta, lon_range=lon_range,
        lat_range=[-5, 5], time_roll=0
    )
    nino4.name = 'nino4'

    lon_range = [130, 160] if antimeridian is False else preproc.get_antimeridian_coord([130, 160])
    nino5, nino5_std = preproc.get_mean_time_series(
        ssta, lon_range=lon_range,
        lat_range=[-5, 5], time_roll=0
    )
    nino5.name = 'nino5'

    nino_idx = xr.merge([nino12, nino3, nino34, nino4, nino5])

    if monthly:
        nino_idx = nino_idx.resample(time='M', label='left' ).mean()
        nino_idx = nino_idx.assign_coords(
            dict(time=nino_idx['time'].data + np.timedelta64(1, 'D'))
        )

    # Cut time period
    if time_range is not None:
        nino_idx = nino_idx.sel(time=slice(
            np.datetime64(time_range[0], "M"), 
            np.datetime64(time_range[1], "M")
        ))
    return nino_idx