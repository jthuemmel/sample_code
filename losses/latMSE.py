import torch as th
import xarray as xr
import glob
import os

class LatMSE(th.nn.Module):
    '''
    Latitude weighted MSE
    lat_weights are calculated from the constants.nc file if the path is provided
    lat_weights are set to 1 (e.g. no weighting) if the path is not provided
    '''
    def __init__(self, data_dir: str = None):
        '''
        :param data_dir: path to the directory where the data files are stored
        '''
        super().__init__()
        
        if data_dir is not None:
            constants = xr.open_mfdataset(glob.glob(os.path.join(data_dir, 'constants', '*.nc')))
            self.lat_weights = th.from_numpy(constants['lat2d'].to_numpy()).float().deg2rad().cos()
        else:
            self.lat_weights = th.Tensor([1])
            
    def forward(self, prediction, target):
        if prediction.device != self.lat_weights.device:
            self.lat_weights = self.lat_weights.to(device = prediction.device)
        diff = (prediction - target).pow(2)
        loss = (self.lat_weights * diff).mean()
        return loss