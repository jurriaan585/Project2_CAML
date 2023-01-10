import numpy as np
import torch

from torch.utils.data import Dataset

class GravitationalWave_datastrain(Dataset):
    def __init__(self, path: str, background = False, args=None ):
        self.path = path     #path to the data folder
        self.args = args     #dummy for later use
        self.background = background
        
        self.length = 15000
        self.array_length = 65536
        
    def get_injection(self,injection_number):
        injection = np.load(self.path+ '/Injections/Whitened/injection_wf'+str(injection_number)+'.npy').T 
        return injection
    def get_background(self,background_number):
        background = np.load(self.path+ '/Background/Whitened/background_wf'+str(background_number)+'.npy').T
        return background
    
    def get_injections(self,injection_numbers):
        injections = np.zeros((len(injection_numbers), 4, self.array_length))
        for i, injection_number in enumerate(injection_numbers):
            injections[i, :, :] = np.load(self.path+ '/Injections/Whitened/injection_wf'+str(injection_number)+'.npy').T
        return injections
    def get_backgrounds(self,background_numbers):
        backgrounds = np.zeros((len(background_numbers),4,self.array_length))
        for i, background_number in enumerate(background_numbers):
            backgrounds[i, :, :] = np.load(self.path+ '/Background/Whitened/background_wf'+str(background_number)+'.npy').T
        return backgrounds
    
    def make_dataset(self, size, data_bounds = [0,15000]):
        random_draw = np.random.randint(data_bounds[0],data_bounds[1], size)
        shuffle_array = np.arange(size*2)
        np.random.shuffle(shuffle_array)

        self.background = True
        backgrounds = np.array(self[list(random_draw)])
        self.background = False
        injections = np.array(self[list(random_draw)])

        truth_values = np.concatenate([np.zeros(size),np.ones(size)])[shuffle_array]
        Truth_values = torch.tensor(truth_values)

        dataset = np.concatenate([injections,backgrounds],1)
        dataset = dataset[:, shuffle_array,:]
        Dataset = torch.tensor(dataset)
        return Dataset, Truth_values

    def __len__(self):
        return self.length         #the total number of injections I can find, is hardcoded for now    
    def __getitem__(self, index):
        if isinstance(index, int):
            if index < 0: #Handle negative indices
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError(f"The index ({index}) is out of range.")
            
            data = self.get_background(index) if self.background else self.get_injection(index)
            E1, E2, E3, time_array = data
                    
        elif isinstance(index, slice):
            indices = np.arange(*(index).indices(len(self)))
            
            data = self.get_backgrounds(indices) if self.background else self.get_injections(indices)
            E1T, E2T, E3T, time_arrayT = data.transpose(1,2,0)
            E1, E2, E3, time_array = E1T.T, E2T.T, E3T.T, time_arrayT.T
        
        elif isinstance(index, list):
            data = self.get_backgrounds(index) if self.background else self.get_injections(index)
            E1T, E2T, E3T, time_arrayT = data.transpose(1,2,0)
            E1, E2, E3, time_array = E1T.T, E2T.T, E3T.T, time_arrayT.T
       
        else:
            raise TypeError("Invalid argument type.")
            
        return E1, E2, E3, time_array

    
    
class GravitationalWave_datastrain_New(Dataset):
    def __init__(self, path: str, background = False, noise_realization = 'Whitened', args=None ):
        self.path = path     #path to the data folder
        self.args = args     #dummy for later use
        self.background = background
        self.length = 15000
        self.array_length = 65536
        self.noise_realization = noise_realization
        
    def get_injection(self,injection_number):
        injection = np.load(self.path+ '/Injections/'+self.noise_realization+'/injection_wf'+str(injection_number)+'.npy.html').T 
        return injection
    def get_background(self,background_number):
        background = np.load(self.path+ '/Background/'+self.noise_realization+'/background_wf'+str(background_number)+'.npy.html').T
        return background
    
    def get_injections(self,injection_numbers):
        injections = np.zeros((len(injection_numbers), 4, self.array_length))
        for i, injection_number in enumerate(injection_numbers):
            injections[i, :, :] = np.load(self.path+ '/Injections/'+self.noise_realization+'/injection_wf'+str(injection_number)+'.npy.html').T
        return injections
    def get_backgrounds(self,background_numbers):
        backgrounds = np.zeros((len(background_numbers),4,self.array_length))
        for i, background_number in enumerate(background_numbers):
            backgrounds[i, :, :] = np.load(self.path+ '/Background/'+self.noise_realization+'/background_wf'+str(background_number)+'.npy.html').T
        return backgrounds
    
    def make_dataset(self, size, data_bounds = [0,15000]):
        random_draw = np.random.randint(data_bounds[0],data_bounds[1], size)
        shuffle_array = np.arange(size*2)
        np.random.shuffle(shuffle_array)

        self.background = True
        backgrounds = np.array(self[list(random_draw)])
        self.background = False
        injections = np.array(self[list(random_draw)])

        truth_values = np.concatenate([np.zeros(size),np.ones(size)])[shuffle_array]
        Truth_values = torch.tensor(truth_values)

        dataset = np.concatenate([injections,backgrounds],1)
        dataset = dataset[:, shuffle_array,:]
        Dataset = torch.tensor(dataset)
        return Dataset, Truth_values

    def __len__(self):
        return self.length         #the total number of injections I can find, is hardcoded for now    
    def __getitem__(self, index):
        if isinstance(index, int):
            if index < 0: #Handle negative indices
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError(f"The index ({index}) is out of range.")
            
            data = self.get_background(index) if self.background else self.get_injection(index)
            E1, E2, E3, time_array = data
                    
        elif isinstance(index, slice):
            indices = np.arange(*(index).indices(len(self)))
            
            data = self.get_backgrounds(indices) if self.background else self.get_injections(indices)
            E1T, E2T, E3T, time_arrayT = data.transpose(1,2,0)
            E1, E2, E3, time_array = E1T.T, E2T.T, E3T.T, time_arrayT.T
        
        elif isinstance(index, list):
            data = self.get_backgrounds(index) if self.background else self.get_injections(index)
            E1T, E2T, E3T, time_arrayT = data.transpose(1,2,0)
            E1, E2, E3, time_array = E1T.T, E2T.T, E3T.T, time_arrayT.T
       
        else:
            raise TypeError("Invalid argument type.")
            
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
        backgrounds = [np.load(self.path+ '/Background/Whitened/background_wf'+str(i)+'.npy').T for i in range(number_of_backgrounds)]
        return backgrounds
    def __getitem__(self, index, background: bool = False ):
        injection = self.background[index] if background else self.injections[index]
        E1, E2, E3, time_array = injection
        return E1, E2, E3, time_array
        