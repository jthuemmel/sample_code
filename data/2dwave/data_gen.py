import torch as th
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

from torch.utils.data import Dataset
from wave_generator import WaveGenerator


def load_attributes(f,name = 'params'):
    '''
    Function to load attributes of a param group in an hdf5 file.
    File structure is assumed to be:
    params/param_types
    The function will return a list of dictionaries containing the attributes of param_types
    '''
    param_list = []
    for group in f[name].keys():
        att = f[f'{name}/{group}'].attrs
        params = {}
        for key in att.keys():
            params[key] = att[key]
        param_list.append(params)
    return param_list

def load_data(f, name = 'data'):
    '''
    Function to load data from an hdf5 file.
    The function assumes that the file has a structure similar to:
    data/datasets/samples
    The function will extract all datasets and all samples for each dataset and return a list of np.arrays
    '''
    data_list = []
    for group in f[name].keys():
        dset = f[f'{name}/{group}']
        data = []
        for key in dset.keys():
            data.append(dset[key][()])
        data_list.append(np.array(data))
    return data_list
    
class WaveData(Dataset):
    def __init__(self, path: str, name: str, group: str = 'train'):
        '''
        A custom class to load data from an hdf5 file.
        The function assumes that the file has a structure similar to:
            data/datasets/samples
            
        :param path: The path of the hdf5 file
        :param name: The name of the dataset
        :param group: The name of the subset (train, val, test) to retrieve
        '''
        assert group in ['train','val','test'], 'Please choose from [train, val, test]'
        self.group = group
        self.data_path = os.path.join(path, name)
               
        self.dataset = h5py.File(self.data_path,'r')
        
        self.num_samples = len(self.dataset[f'data/{self.group}'])
        self.keys = list(self.dataset[f'data/{self.group}'].keys())
        
        self.dataset.close()
        self.dataset = None       
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self,idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.data_path, 'r')
        data = self.dataset[f'data/{self.group}'][self.keys[idx]][:]
        return data    

if __name__ == 'main':
    
    dataset_name = 'small2DWave'
    
    sim_parameters = {
        'dt' : 0.1,
        'dx' : 1,
        'dy' : 1,
        "timesteps" : 100,
        'width' : 32,
        'height' : 32
    }

    wave_parameters = {
         'amplitude': 0.34,
         'velocity': 3.0,
         'damp': 1.,
         'width_x': .5,
         'width_y': .5,
         'skip_rate': 1,
         'waves': 1,
         'velocities': [0.5,5.0],
         'amplitudes': [0.1, 0.34],
         'skip_rates': [1,5],
         'p_absorption': 0.0,
         'p_generation': 0.0,
         'std_absorption': 0.01,
         'std_generation': 0.01,
         'obstacles_N': 0,
         'obstacles_dim': [],
         'obstacles_given': [((5,8),(6,1))],
         'boundary_type':'reflecting'
    }

    data_parameters = {
        'samples_train' : 10000,
        'samples_val' : 100,
        'samples_test' : 100,
        'name' : dataset_name,
        'save_data' : True
    }

    wave_generator = WaveGenerator(
        dt=sim_parameters["dt"],
        dx=sim_parameters["dx"],
        dy=sim_parameters["dy"],
        timesteps=sim_parameters["timesteps"],
        width=sim_parameters["width"],
        height=sim_parameters["height"],
        amplitude=wave_parameters["amplitude"],
        velocity=wave_parameters["velocity"],
        #velocity_range=wave_parameters['velocities'],
        damp=wave_parameters["damp"],
        wave_width_x=wave_parameters["width_x"],
        wave_width_y=wave_parameters["width_y"],
        p_absorption=wave_parameters["p_absorption"],
        p_generation=wave_parameters["p_generation"],
        std_absorption=wave_parameters["std_absorption"],
        std_generation=wave_parameters["std_generation"],
        waves =wave_parameters["waves"],
        boundary_type = wave_parameters["boundary_type"],
        obstacles_N = wave_parameters["obstacles_N"],
        obstacles_dim = wave_parameters['obstacles_dim'],
        obstacles_given = wave_parameters['obstacles_given']
        )



    if not os.path.exists(data_parameters['name']) and data_parameters['save_data']:
        with h5py.File(data_parameters['name'],'a') as f:
            #create training data
            for i in range(data_parameters['samples_train']):
                sample = wave_generator.generate_sample()
                f.require_dataset(name=f'data/train/{i:03}',data=sample, shape = sample.shape, dtype = sample.dtype)
            #create validation data
            for i in range(data_parameters['samples_val']):
                sample = wave_generator.generate_sample()
                f.require_dataset(name=f'data/val/{i:03}',data=sample, shape = sample.shape, dtype = sample.dtype) 
            #create test data
            for i in range(data_parameters['samples_test']):
                sample = wave_generator.generate_sample()
                f.require_dataset(name=f'data/test/{i:03}',data=sample, shape = sample.shape, dtype = sample.dtype)  

            f.require_group('params/wave_params')
            f.require_group('params/sim_params')

            for key,value in wave_parameters.items():
                f['params/wave_params'].attrs[key] = value
            for key,value in sim_parameters.items():
                f['params/sim_params'].attrs[key] = value