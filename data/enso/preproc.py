''' Collection of functions for preprocessing data.

@Author  :   Jakob Schlör 
@Time    :   2023/05/09 13:20:26
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import numpy as np
import xarray as xr
import cftime

def check_dimensions(ds, sort=True, verbose = False):
    """
    Checks whether the dimensions are the correct ones for xarray!
    """
    dims = list(ds.dims)

    rename_dic = {
        'longitude': 'lon',
        'latitude': 'lat',
        'nav_lon': 'lon',
        'nav_lat': 'lat'
    }
    for c_old, c_new in rename_dic.items():
        if c_old in dims:
            if verbose:
                print(f'Rename:{c_old} : {c_new} ', flush=True)
            ds = ds.rename({c_old: c_new})
            dims = list(ds.dims)

    # Check for dimensions
    clim_dims = ['lat', 'lon']
    for dim in clim_dims:
        if dim not in dims:
            raise ValueError(
                f"The dimension {dim} not consistent with required dims {clim_dims}!")

    # If lon from 0 to 360 shift to -180 to 180
    if max(ds.lon) > 180:
        if verbose:
            print("Shift longitude!", flush=True)
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))

    if sort:
        if verbose:
            print('Sort longitudes and latitudes in ascending order, respectively', flush=True)
        ds = ds.sortby('lon')
        ds = ds.sortby('lat')

    return ds

def time_average(ds, group='1D'):
    """Downsampling of time dimension by averaging.

    Args:
    -----
    ds: xr.dataFrame 
        dataset
    group: str
        time group e.g. '1D' for daily average from hourly data
    """
    ds_average = ds.resample(time=group, label='left').mean(skipna=True)

    # Shift time to first of month
    if group == '1M':
        new_time = ds_average.time.data + np.timedelta64(1, 'D')
        new_coords = {}
        for dim in ds_average.dims:
            new_coords[dim] = ds_average[dim].data
        new_coords['time'] = new_time
        ds_average = ds_average.assign_coords(coords=new_coords)

    return ds_average


def set_grid(ds, step_lat=1, step_lon=1,
             lat_range=None, lon_range=None):
    """Interpolate grid.

    Args:
        ds (xr.Dataset): Dataset or dataarray to interpolate.
            Dataset is only supported for grid_type='mercato'.
        step_lat (float, optional): Latitude grid step. Defaults to 1.
        step_lon (float, optional): Longitude grid step. Defaults to 1.

    Returns:
        da (xr.Dataset): Interpolated dataset or dataarray.
        grid (dict): Grid used for interpolation, dict(lat=[...], lon=[...]).
    """
    lat_min = ds['lat'].min().data if lat_range is None else lat_range[0] 
    lat_max = ds['lat'].max().data if lat_range is None else lat_range[1] 
    lon_min = ds['lon'].min().data if lon_range is None else lon_range[0] 
    lon_max = ds['lon'].max().data if lon_range is None else lon_range[1] 
    init_lat = np.arange(
        lat_min, (lat_max + step_lat), step_lat
    )
    init_lon = np.arange(
        lon_min, lon_max, step_lon
    )
    grid = {'lat': init_lat, 'lon': init_lon}
    # Interpolate
    da = ds.interp(grid, method='linear')

    return da, grid


def cut_map(ds, lon_range=None, lat_range=None, shortest=True):
    """Cut area. 
    
    Args:
        da (xr.Dataset): 
        lon_range (list): Range of longitudes [min, max]. Defaults to None.
        lat_range (list): Range of latitudes [min, max]. Defaults to None.
        shortest (boolean): Use shortest range in longitude, 
            eg. -170, 170 range contains all points from 170 to 180, -180 to -170,
            and not all between -170 and 170. Default is True.
    Return:
        ds (xr.Dataset): Dataset cut to range
    """
    if lon_range is not None:
        if (max(lon_range) - min(lon_range) <= 180) or shortest is False:
            ds = ds.sel(
                lon=slice(np.min(lon_range), np.max(lon_range)),
                lat=slice(np.min(lat_range), np.max(lat_range))
            )
        else:
            # To account for areas that lay at the border of -180 to 180
            ds = ds.sel(
                lon=ds.lon[(ds.lon < min(lon_range)) |
                           (ds.lon > max(lon_range))],
                lat=slice(np.min(lat_range), np.max(lat_range))
            )
    if lat_range is not None:
        ds = ds.sel(
            lat=slice(np.min(lat_range), np.max(lat_range))
        )

    return ds


def get_antimeridian_coord(lons, verbose = False):
    """Change of coordinates from normal to antimeridian."""
    lons = np.array(lons)
    lons_new = np.where(lons < 0, (lons + 180), (lons - 180))
    return lons_new


def set_antimeridian2zero(ds, roll=True, verbose = False):
    """Set the antimeridian to zero.

    Easier to work with the pacific then.
    """
    if ds['lon'].data[0] <= -100 and roll is True:
        # Roll data such that the dateline is not at the corner of the dataset
        if verbose:
            print("Roll longitudes.", flush=True)
        ds = ds.roll(lon=(len(ds['lon']) // 2), roll_coords=True)

    # Change lon coordinates
    lons_new = get_antimeridian_coord(ds.lon)
    ds = ds.assign_coords(
        lon=lons_new
    )
    if verbose:
        print('Set the dateline to the new longitude zero.', flush=True)
    return ds


def compute_anomalies(dataarray, group='dayofyear',
                      base_period=None):
    """Calculate anomalies.

    Parameters:
    -----
    dataarray: xr.DataArray
        Dataarray to compute anomalies from.
    group: str
        time group the anomalies are calculated over, i.e. 'month', 'day', 'dayofyear'
    base_period (list, None): period to calculate climatology over. Default None.

    Return:
    -------
    anomalies: xr.dataarray
    """
    if base_period is None:
        base_period = np.array(
            [dataarray.time.data.min(), dataarray.time.data.max()])

    climatology = dataarray.sel(time=slice(base_period[0], base_period[1])
                                ).groupby(f'time.{group}').mean(dim='time', skipna=True)
    anomalies = (dataarray.groupby(f"time.{group}")
                 - climatology)

    return anomalies


def detrend_dim(da, dim='time', deg=1, startyear=None):
    """Detrend data by subtracting a linear fit which is obtained
        for the lat-lon mean at each time.

    Args:
        da ([xr.DataArray]): Data to detrend.

    Returns:
        da_detrend (xr.DataArray): Detrended data.
        coef (list): Linear fit coefficients,
                    i.e. coef[0]*x + coef[1]
    """
    if startyear is not None:
        tmin, tmax = da.time.data.min(), da.time.data.max()
        ttrend = np.datetime64(f'{startyear}-01-01', 'D') -1
        da_notrend = da.sel(time=slice(tmin, ttrend))
        da_trend = da.sel(time=slice(ttrend, tmax))
    else:
        da_trend = da
    
    # Linear fit to data 
    p = da_trend.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da_trend[dim], p.polyfit_coefficients)
    da_detrend =  da_trend - fit + fit[0]

    if startyear is not None:
        da_detrend = xr.concat([da_notrend, da_detrend], dim='time')

    return da_detrend

def get_mean_time_series(da, lon_range, lat_range, time_roll=0):
    """Get mean time series of selected area.

    Parameters:
    -----------
    da: xr.DataArray
        Data
    lon_range: list
        [min, max] of longitudinal range
    lat_range: list
        [min, max] of latiduninal range
    """
    da_area = cut_map(da, lon_range, lat_range)
    ts_mean = da_area.mean(dim=('lon', 'lat'), skipna=True)
    ts_std = da_area.std(dim=('lon', 'lat'), skipna=True)
    if time_roll > 0:
        ts_mean = ts_mean.rolling(time=time_roll, center=True).mean()
        ts_std = ts_std.rolling(time=time_roll, center=True).mean()

    return ts_mean, ts_std


def is_datetime360(time):
    return isinstance(time, cftime._cftime.Datetime360Day)

def is_datetime(time):
    return isinstance(time, cftime._cftime.DatetimeNoLeap)

def time2timestamp(time):
    """Convert time to timestamp."""
    try:
        t = time[0]
    except:
        t = time

    if is_datetime(t):
        timestamp = cftime.date2num(t, 'days since 0001-01-01')
    else:
        timestamp = (t - np.datetime64('0001-01-01', 'ns')) / np.timedelta64(1, 'D') 
    return timestamp

def timestamp2time(timestamp, type=np.ndarray):
    """Convert timestamp to np.datetime64 object."""
    if type == cftime._cftime.DatetimeNoLeap:
        time = cftime.num2date(timestamp, 'days since 0001-01-01')
    else:
        time = (np.datetime64('0001-01-01', 'ns') 
                + timestamp * np.timedelta64(1, 'D') )
    return time

def time2idx(timearray, times):
    """Get index of t in times.
    
    Parameters:
        timearray (np.ndarray): Time array to search in.
        times (list): List of times.
    """
    idx = []
    for t in times:
        idx.append(np.argmin(np.abs(timearray - t)))
    return idx


def convert_datetime64(date: np.datetime64, dtype=cftime._cftime.DatetimeNoLeap):
    """Convert datetime64 dates into other time formats.

    Args:
        date (np.datetime64): Input date
        dtype (_type_, optional): Date type format.
            Defaults to cftime._cftime.DatetimeNoLeap.

    Returns:
        _type_: Date in other format.
    """
    date = np.array(date, dtype='datetime64[D]')
    years = date.astype('datetime64[Y]').astype(int) + 1970
    months = date.astype('datetime64[M]').astype(int) % 12 + 1
    days = (date - date.astype('datetime64[M]') + 1).astype(int)

    if date.size == 1:
        dates = cftime._cftime.DatetimeNoLeap(years, months, days)
    else:
        dates = []
        for i in range(date.size):
            if dtype == cftime._cftime.DatetimeNoLeap:
                dates.append(cftime._cftime.DatetimeNoLeap(years[i], months[i], days[i]))
    return np.array(dates)

def map2flatten(x_map: xr.Dataset) -> list:
    """Flatten dataset/dataarray and remove NaNs.

    Args:
        x_map (xr.Dataset/ xr.DataArray): Dataset or DataArray to flatten. 

    Returns:
        x_flat (xr.DataArray): Flattened dataarray without NaNs 
        ids_notNaN (xr.DataArray): Boolean array where values are on the grid. 
    """
    if type(x_map) == xr.core.dataset.Dataset:
        x_stack_vars = [x_map[var] for var in list(x_map.data_vars)]
        x_stack_vars = xr.concat(x_stack_vars, dim='var')
        x_stack_vars = x_stack_vars.assign_coords({'var': list(x_map.data_vars)})
        x_flatten = x_stack_vars.stack(z=('var', 'lat', 'lon')) 
    else:
        x_flatten = x_map.stack(z=('lat', 'lon'))

    # Flatten and remove NaNs
    if 'time' in x_flatten.dims:
        idx_notNaN = ~np.isnan(x_flatten.isel(time=0))
    else:
        idx_notNaN = ~np.isnan(x_flatten)
    x_proc = x_flatten.isel(z=idx_notNaN.data)

    return x_proc, idx_notNaN


def flattened2map(x_flat: np.ndarray, ids_notNaN: xr.DataArray, times: np.ndarray = None) -> xr.Dataset:
    """Transform flattened array without NaNs to gridded data with NaNs. 

    Args:
        x_flat (np.ndarray): Flattened array of size (n_times, n_points) or (n_points).
        ids_notNaN (xr.DataArray): Boolean dataarray of size (n_points).
        times (np.ndarray): Time coordinate of xarray if x_flat has time dimension.

    Returns:
        xr.Dataset: Gridded data.
    """
    if len(x_flat.shape) == 1:
        x_map = xr.full_like(ids_notNaN, np.nan, dtype=float)
        x_map[ids_notNaN.data] = x_flat
    else:
        temp = np.ones((x_flat.shape[0], ids_notNaN.shape[0])) * np.nan
        temp[:, ids_notNaN.data] = x_flat
        if times is None:
            times = np.arange(x_flat.shape[0]) 
        x_map = xr.DataArray(data=temp, coords={'time': times, 'z': ids_notNaN['z']})

    if 'var' in list(x_map.get_index('z').names):
        x_map = x_map.unstack()

        if 'var' in list(x_map.dims): # For xr.Datasset only
            da_list = [xr.DataArray(x_map.isel(var=i), name=var) 
                       for i, var in enumerate(x_map['var'].data)]
            x_map = xr.merge(da_list, compat='override')
            x_map = x_map.drop(('var'))
    else:
        x_map = x_map.unstack()
    
    return x_map




def process_data(f_data, vars, 
              time_average=None, antimeridian=False,
              lon_range=None, lat_range=None, 
              climatology=None, detrend_from=None,
              normalization=None, splity=None,
              grid_step=None, mask=None, verbose = False):
    """Load and preprocess data using xarray.

    Args:
        f_sst (str): Filename.
        vars (list): List of variable names.
        time_average (str, optional): Resample time and average, e.g. 'M'. 
            Defaults to None.
        antimeridian (bool, optional): Set the antimeridian to zero.
            Defaults to False.
        lon_range (list, optional): Longitude range to cut, e.g. [90, 120].
            Defaults to None.
        lat_range (list, optional): Latitude range to cut, e,g, [-10, 10]. 
            Defaults to None.
        grid_step (float, optional): Grid step for interpolation.
            Defaults to None.
        climatology (str, optional): If set anomalies are computed with respect to the 
            choosen climatology, e.g. 'day', 'month'. Defaults to None.
        normalization (str, optional): Normalize data by either 'minmax' or 'zscore'.
            Defaults to None.
        splity (int, optional): Year to split dataarray into training and test set. 
            Defaults to None.

    Returns:
        da (xr.DataArray): Processed dataarray.
        da_train (xr.DataArray): Processed dataarray until splity.
            Not returned if splity=None
        da_test (xr.DataArray): Processed dataarray from splity to end.
            Not returned if splity=None
    """
    ds = xr.open_dataset(f_data)
    ds = check_dimensions(ds)

    da_preprocessed = []
    for varname in vars:
        if verbose:
            print(f"Preprocess {varname}:", flush=True)
        da = ds[varname]

        if mask is not None:
            if verbose:
                print('Apply mask.', flush=True)
            da = da.where(mask == 0.0)

        if time_average is not None:
            if verbose:
                print(f"Resample time by {time_average} and compute mean.", flush=True)
            da = da.resample(time=time_average, label='left').mean()
            da = da.assign_coords(
                dict(time=da['time'].data + np.timedelta64(1, 'D'))
            )

        # change coordinates to dateline == 0
        if antimeridian:
            da = set_antimeridian2zero(da)
            if lon_range is not None:
                lon_range = get_antimeridian_coord(lon_range)
        
        # Cut area of interest
        if lon_range is not None or lat_range is not None:
            if verbose:
                print(f'Get selected area: lon={lon_range}, lat={lat_range}!', flush=True)
            da = cut_map(
                da, lon_range=lon_range, lat_range=lat_range, shortest=False
            )

        # coarse grid if needed
        if grid_step is not None:
            if verbose:
                print(f'Interpolate grid on res {grid_step}', flush=True)
            da, grid = set_grid(da, step_lat=grid_step, step_lon=grid_step,
                                lat_range=lat_range, lon_range=lon_range)
        
        # Detrend
        if detrend_from is not None:
            if verbose:
                print("Detrend data:", flush=True)
            da = detrend_dim(da, dim='time', startyear=detrend_from)

        # Compute anomalies
        if climatology is not None:
            if verbose:
                print("Compute anomalies:", flush=True)
            da = compute_anomalies(da, group=climatology)
            da.name = f"{varname}a"

        # Normalize data
        if normalization is not None:
            normalizer = Normalizer(method=normalization)
            da = normalizer.fit_transform(da)
            # Store normalizer as an attribute in the Dataarray for the inverse
            # transformation
            da.attrs = {'normalizer': normalizer}
        
        da_preprocessed.append(da)
    
    ds_preproc = xr.merge(da_preprocessed)
    
    # Split into training and test
    if splity is None:
        return ds_preproc

    else:
        start_time = ds_preproc.time.min().data
        end_time = ds_preproc.time.max().data
        split_time = np.datetime64(f"{splity}-01-01", 'D')
        ds_train = ds_preproc.sel(time=slice(start_time, split_time))    
        ds_test = ds_preproc.sel(time=slice(split_time, end_time))    
        return ds_preproc, ds_train, ds_test


class Normalizer():
    """Normalizing xr.DataArray.

    Args:
        method (str, optional): Normalization method, i.e.
            'minmax': Normalize data between [0,1], 
            'zscore': Standardizes the data,  
            'center': Centers the data around 0,  
            Defaults to 'zscore'.

    Raises:
        ValueError: If method is none of the listed above. 
    """
    
    def __init__(self, method: str = 'zscore') -> None:
        self.method = method

        if self.method not in ['zscore', 'minmax', 'center']:
            raise ValueError(f"Your selected normalization method "+
                             self.method + " does not exist.", flush=True)


    def fit(self, da: xr.DataArray, axis: int = None, dim: str = None, **kwargs) -> None:
        """Compute the parameters of the normalization for the given data.

        Args:
            da (xr.DataArray): Data to compute the normalization for.
            dim (str, optional): Dimension along the normalization should be performed.
                Defaults to None.
            axis (int, optional): Axis along the normalization should be performed. If
                dim is specified, axis will not be used. Defaults to None.
        """
        if dim is None and axis is not None:
            dim = da.dims[axis]
        elif dim is None and axis is None:
            dim = da.dims

        if self.method == 'minmax':
            self.min = da.min(dim=dim, skipna=True)
            self.max = da.max(dim=dim, skipna=True)

        elif self.method == 'zscore':
            self.mean = da.mean(dim=dim, skipna=True)
            self.std = da.std(dim=dim, skipna=True)

        elif self.method == 'center':
            self.mean = da.mean(dim=dim, skipna=True)

        return None
    

    def transform(self, da: xr.DataArray, **kwargs) -> xr.DataArray:
        """Normalize data using the normalization parameters.

        Args:
            da (xr.DataArray): Data to compute the normalization for.
        
        Returns:
            xr.DataArray: Normalized data. 
        """
        if self.method == 'minmax':
            return (da - self.min) / (self.max - self.min)

        elif self.method == 'zscore':
            return (da - self.mean) / self.std

        elif self.method == 'center':
            return da - self.mean 
    

    def fit_transform(self, da: xr.DataArray, axis: int = None,
                      dim: str = None, **kwargs) -> xr.DataArray:
        """Compute the normalization parameters and transform the data.

        Args:
            da (xr.DataArray): Data to compute the normalization for.
            dim (str, optional): Dimension along the normalization should be performed.
                Defaults to None.
            axis (int, optional): Axis along the normalization should be performed. If
                dim is specified, axis will not be used. Defaults to None.

        Returns:
            xr.DataArray: Normalized data. 
        """
        self.fit(da, axis=axis, dim=dim, **kwargs)
        da_norm = self.transform(da, **kwargs)
        return da_norm
    

    def inverse_transform(self, da_norm: xr.DataArray) -> xr.DataArray:
        """Inverse the normalization.

        Args:
            da_norm (xr.DataArray): Normalized data. 

        Returns:
            xr.DataArray: Unnormalized data.
        """
        if self.method == 'minmax':
            return da_norm * (self.max - self.min) + self.min
        elif self.method == 'zscore':
            return da_norm * self.std + self.mean
        elif self.method == 'center':
            return da_norm + self.mean