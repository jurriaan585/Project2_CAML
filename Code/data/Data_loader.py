import numpy as np
import torch

from torch.utils.data import Dataset

class GravitationalWave_datastrain(Dataset):
    def __init__(self, path: str, args=None, background: bool = False ):
        self.path = path     #path to the data folder
        self.args = args     #dummy for later use
        self.backgroud = background
        
    def get_injection(self,injection_number):
        injections = np.load(self.path+ '/Injections/Whitened/injection_wf'+str(injection_number)+'.npy') 
        return injections
    def get_background(self,background_number):
        backgrounds = np.load(self.path+ '/Background/Whitened/background_wf'+str(background_number)+'.npy')
        return backgrounds
    
    def __len__(self):
        return 10000         #the total number of injections I can find, is hardcoded for now
    
    def __getitem__(self, index):
        injection = self.get_background(index) if self.background else self.get_injection(index)
        E1, E2, E3, time_array = injection.T
        return E1, E2, E3, time_array


class GravitationalWave_datastrain_MEMORYINTENSIVE(Dataset):
    def __init__(self, path: str, args=None):
        self.path = path     #path to the data folder
        self.args = args     #dummy for later use
        print('Loading Injections:')
        num=100             #this is now hardcoded because IDK how many samples there are in the folder
        self.injections = self.get_injections(num)
        print('Injections loaded.')
        print('Loading Background:')
        self.background = self.get_background(num)
        print('Background loaded.')
        
    def get_injections(self,number_of_injections):
        injections = [np.load(self.path+ '/Injections/Whitened/injection_wf'+str(i)+'.npy').T for i in range(number_of_injections)]
        return injections
    def get_background(self,number_of_backgrounds):
        backgrounds = [np.load(self.path+ '/Background/Whitened/background_wf'+str(i)+'.npy.html').T for i in range(number_of_backgrounds)]
        return backgrounds
    def __getitem__(self, index, background: bool = False ):
        injection = self.background[index] if background else self.injections[index]
        E1, E2, E3, time_array = injection
        return E1, E2, E3, time_array
        