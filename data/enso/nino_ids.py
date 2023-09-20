import numpy as np
import xarray as xr
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from stfor.utils import preproc, metric

class NinoIDs(Dataset):
    def __init__(
        self, 
        datapath: list,
        f_lsm: str,
        lon_range: list = [130, -70], 
        lat_range: list = [-31, 32], 
        grid_step: float = 1.,
        climatology: str = 'month', 
        time_period: slice = slice('0000', '1200'),
        history: int = 4, 
        horizon: int = 24, 
        normalization: str = 'zscore',
        antimeridian: bool = True,
        **kwargs):
        super().__init__()

        # load land sea mask
        land_area_mask = preproc.process_data(
            f_lsm, ['sftlf'],
            antimeridian=True, lon_range=lon_range, lat_range=lat_range,
            grid_step=grid_step,
        )['sftlf']

        # load data
        da_arr = []
        for f in datapath:
            da = preproc.process_data(
                f['path'], [f['vname']], antimeridian=antimeridian, lon_range=lon_range, 
                lat_range=lat_range, climatology=climatology, detrend_from=None,
                grid_step=grid_step, normalization=normalization, verbose = False
            )[f"{f['vname']}a"]
            da = da.where(land_area_mask == 0.0)
            da_arr.append(da)
        # merge data
        ds = xr.merge(da_arr)

        #compute nino indices
        x_target = ds.normalizer.inverse_transform(ds['tsa'])
        ids = metric.get_nino_indices(x_target, antimeridian=True)

        #fill NaN values with zero
        ds = ds.fillna(0)

        # select time period
        self.data = ds.sel(time=time_period)
        self.ids = ids.sel(time=time_period)
        
        #attributes
        self.history = history
        self.horizon = horizon
        self.normalization = normalization
        #land-sea mask
        self.lsm = torch.logical_not(torch.from_numpy(land_area_mask.where(land_area_mask == 0, 1).data))

    def __len__(self):
        return len(self.data['time']) - self.history - self.horizon
    
    def __getitem__(self, idx):
        nino_ids = torch.FloatTensor(self.ids.isel(time=slice(idx, idx+self.history+self.horizon)).to_array().values)
        x = torch.FloatTensor(self.data.isel(time=slice(idx, idx+self.history+self.horizon)).to_array().values)
        return x, nino_ids