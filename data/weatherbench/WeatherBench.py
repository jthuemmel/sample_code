import glob
import os
from pathlib import Path

import torch as th
import xarray as xr


class WeatherBench(th.utils.data.Dataset):
  '''
  Dataloader for the WeatherBench dataset
  :param context: The number of time steps to use as input.
  :param horizon: The number of time steps to use as target.
  :param src_path: The path where the data folders are located
  :param time_period: Time-period to run the data over (e.g. slice(1979,2015) for training data)
  :param predicted_var: Dynamic variables to predict.
  :param prescribed_var: Dynamic variables to use as input, but not predict.
  :param static_var: Static variables to include.
  :param rate: The temporal sampling rate (e.g. 6 to return every 6th hour)
  '''

  def __init__(self,
               context: int,
               horizon: int,
               src_path: str = 'WeatherBench',
               time_period: slice = slice('1979', '2015', 1),
               predicted_var: list = ['t850', 'z500'],  # 'z250', 'z500', 'z1000', 't2m'
               prescribed_var: list = [],  # 'tisr'
               static_var: list = [],  # 'lat2d', 'lon2d', 'lsm', 'orography'
               rate: int = 1,
               **kwargs):
    self.tau = (context + horizon) * rate
    self.rate = rate
    time_period = slice(*time_period)
    predicted_var = list(predicted_var)
    prescribed_var = list(prescribed_var)
    static_var = list(static_var)
    # load, filter, normalize
    data = self.load_raw(src_path, time_period)  # load nc files
    self.x_data, self.s_data = self.filter_data(data, time_period, predicted_var, prescribed_var, static_var)
    self.normalize_data()

    # the lists of variable names are used for indexing later
    self.prescribed_var = prescribed_var
    self.predicted_var = predicted_var
    self.static_var = static_var

    # create attributes for outside access
    self.latitude = self.x_data.lat
    self.longitude = self.x_data.lon
    self.time = self.x_data.time
    self.context = context
    self.horizon = horizon

  def load_raw(self, src_path, time_period):
    print('Loading data...')
    start = int(time_period.start) if time_period.start else 1900
    stop = int(time_period.stop) if time_period.stop else 2100
    step = time_period.step if time_period.step else 1
    files = []
    for file in Path(src_path).glob('*/*_[0-9][0-9][0-9][0-9]_*.nc'):
      year = int(file.name.split('_')[-2])
      if year in range(start, stop, step):
        files += [file]

    if len(files) == 0:
      raise ValueError('Can not find data files. Is the dataset path and your slice correct?')

    files += Path(src_path).glob('constants/*.nc')

    return xr.open_mfdataset(files, parallel=False)

  def filter_data(self, data, time_period, predicted_var, prescribed_var, static_var):
    # select the time period we want
    data = data.sel(time=time_period)
    # remove the level coordinate by moving it to individual variables
    for var in list(data.keys()):
      if 'level' in data[var].coords:
        for l in data['level'].values:
          data[f'{var}{l}'] = data.sel(level=l)[var]
    data = data.drop_dims('level')
    # select the variables we care about
    x_data = data[predicted_var + prescribed_var]
    s_data = data[static_var]

    return x_data, s_data

  def normalize_data(self):
    self.mean = self.x_data.mean(('time', 'lat', 'lon'))
    self.std = self.x_data.std('time').mean(('lat', 'lon'))
    self.x_data = ((self.x_data - self.mean) / self.std)
    # constant fields are re-scaled to [-1, 1]
    s_min, s_max = self.s_data.min(), self.s_data.max()
    self.s_data = 2*((self.s_data - s_min) / (s_max - s_min)) - 1
    print('Preprocessing dynamic fields...')
    self.x_data = self.x_data.compute()
    print('Preprocessing static fields...')
    self.s_data = self.s_data.compute()
    print('Done')

  def __len__(self):
    return len(self.x_data.time) - self.tau

  def __getitem__(self, idx):
    # Predictive variables
    predictive = self.x_data[self.predicted_var].isel(
        time=slice(idx, idx + self.tau, self.rate)
    ).to_array().values
    # Prescribed variables
    if len(self.prescribed_var) > 0 and len(self.static_var) > 0:
      prescribed = xr.merge([
          self.x_data[self.prescribed_var].isel(time=slice(idx, idx + self.tau, self.rate)),
          self.s_data[self.static_var]
      ]).to_array().values
    elif len(self.static_var) > 0:
      prescribed = self.s_data[self.static_var].expand_dims(
          dim={'time': self.tau // self.rate}).to_array().values
    elif len(self.prescribed_var) > 0:
      prescribed = self.x_data[self.prescribed_var].isel(
          time=slice(idx, idx + self.tau, self.rate)).to_array().values
    else:
      prescribed = None
    # returning (x, None) crashes the PyTorch dataloader :(
    if prescribed is not None:
      return predictive, prescribed
    else:
      return predictive
