import os
import h5py
from torch.utils.data import Dataset

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