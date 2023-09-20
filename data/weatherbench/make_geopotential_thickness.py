import xarray as xr
import numpy as np
import os 
import glob

if __name__ == 'main':

    src_path = 'WeatherBench'
    paths = [gg for folder in ['geopotential', 'temperature']
                     for gg in glob.glob(os.path.join(src_path, folder, '*.nc'))]  

    new_path = os.path.join(src_path,'geopotential_thickness_300_700')
    if not os.path.exists(new_path):
        os.mkdir(new_path)

    with xr.open_mfdataset(paths, parallel = True) as data:
        for var in list(data.keys()):
            if 'level' in data[var].coords:
                for l in data['level'].values:
                    data[f'{var}{l}'] = data.sel(level = l)[var]
        data = data.drop_dims('level')
        R = 8.3144621 #idealised gas constant
        temps = ['t300','t400','t500','t600','t700'] #available temperature levels between 300 and 700 hPa
        data['tau_300_700'] = R * np.log(data['z300'] / data['z700']) * (sum(data[t] for t in temps)/len(temps))
        #write output files
        for year in np.arange(1979,2019):
            thickness = data['tau_300_700'].sel(time = f'{year}')
            thickness.to_netcdf(os.path.join(new_path, f'geopotential_thickness_{year}_5.625deg.nc'))