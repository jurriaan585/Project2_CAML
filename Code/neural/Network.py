import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """ Generates a CNN """
    def __init__(self):
        """"""
        super(Net, self).__init__()
          
        self.conv1 = nn.Conv2d(1, 1, 8, bias=False, dtype = torch.float32)
        self.pool = nn.MaxPool2d(8,256)
        
        #self.batchnorm = nn.BatchNorm2d(1)
        self.linear1 = nn.Linear(192, 32)
        self.linear2 = nn.Linear(32,2)


    def set_kernels(self,kernels):
        with torch.no_grad():
            self.conv1.weight = torch.nn.Parameter(kernels.unsqueeze(1))

    def forward(self, x):
        #print('0:',x.shape)
        with torch.no_grad():
            x = self.conv1(x)
            #print('1:', x.shape)
        x = self.pool(F.relu(x.squeeze(2)))
        #print('2:', x.shape)
        x = F.relu(self.linear1(x))
        #print('3:', x.shape)
        #x = F.gelu(self.linear2(x))
        #print('4:', x.shape)
        x = F.normalize(self.linear2(x),dim=2)
        #print(x)
        #print('5:', x.shape)
        return x

class Net2(nn.Module):
    """ Generates a CNN """
    def __init__(self):
        """"""
        super(Net2, self).__init__()
          
        self.conv1 = nn.Conv2d(1, 1, 8, bias=False, dtype = torch.float32)
        self.pool = nn.MaxPool2d(8,256)
        
        #self.batchnorm = nn.BatchNorm2d(1)
        self.linear1 = nn.Linear(192, 32)
        self.linear2 = nn.Linear(32,2)


    def set_kernels(self,kernels):
        self.conv1.weight = torch.nn.Parameter(kernels.unsqueeze(1))

    def forward(self, x):
        #print('0:',x.shape)
      
        x = self.conv1(x)
        #print('1:', x.shape)
        x = self.pool(F.relu(x.squeeze(2)))
        #print('2:', x.shape)
        x = F.relu(self.linear1(x))
        #print('3:', x.shape)
        #x = F.gelu(self.linear2(x))
        #print('4:', x.shape)
        x = F.normalize(self.linear2(x),dim=2)
        #print(x)
        #print('5:', x.shape)
        return x

    
    
class Net3(nn.Module):
    """ Generates a CNN """
    def __init__(self,data_len,kernel_len):
        """"""
        super(Net3, self).__init__()

        length = int(data_len - kernel_len + 1)
        self.conv1 = nn.Conv1d(3, 3, kernel_size = (length,), bias=False, groups=3)
        
       
        self.linear1 = nn.Linear(length, 256)
        self.linear2 = nn.Linear(256, 32)
        self.linear3 = nn.Linear(32,2)


    def add(self, layer):
        return layer.sum(1).type(torch.float32)

    def set_kernel(self,kernels):
        with torch.no_grad():
            self.conv1.weight = torch.nn.Parameter(kernels)

    def forward(self, x):
        #print('0:',x.shape)
        with torch.no_grad():
            x = self.conv1(x)
            #print('1:', x.shape)
        x = self.add(x)
        #print('2:', x.shape)
        x = F.relu(self.linear1(x))
        #print('3:', x.shape)
        x = F.gelu(self.linear2(x))
        #print('4:', x.shape)
        x = F.normalize(self.linear3(x))
        #print(x)
        #print('5:', x.shape)
        return x
    
    
from .wavenet_model import *

class Net4(nn.Module):
    """ Generates a CNN """
    def __init__(self,data_len,kernel_len):
        """"""
        super(Net4, self).__init__()
        
        length = int(data_len - kernel_len + 1)
        self.conv0 = nn.Conv1d(3, 3, kernel_size = (length,), bias=False, groups=3)
        self.conv1 = nn.Conv1d(3, 64, kernel_size = 3)
        self.conv2 = nn.Conv1d(64,64, kernel_size = 3)
        self.conv3 = nn.Conv1d(64,128, kernel_size = 1)
        self.conv4 = nn.Conv1d(128, 64, kernel_size = 3)
        self.conv5 = nn.Conv1d(64, 1, kernel_size = 16)
        
        self.dropout = nn.Dropout(0.5)
        
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(128)
        
        self.maxpool = nn.MaxPool1d(3)
        self.maxpool2 = nn.MaxPool1d(2)
        
        self.linear1 = nn.Linear(2722, 256)
        self.linear2 = nn.Linear(256, 32)
        self.linear3 = nn.Linear(32,2)

    def set_kernel(self,kernels):
        self.conv0.weight = torch.nn.Parameter(kernels)
        
    def forward(self, x):
        x = self.conv0(x)
        
        #print('0:',x.shape)
        x = self.conv1(x)
        #x = self.batchnorm1(x)
        x = F.relu(x)
        #print('1:', x.shape)
        
        
        x = self.conv2(x)
        #x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        #print('2:', x.shape)
        
        
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        
        x = self.conv4(x)
        #x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        #print('3:', x.shape)
        
        x = self.conv5(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        #print('4:', x.shape)
        
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        #print('5:', x.shape)
        x = F.relu(self.linear2(x))
        #print('6:', x.shape)
        x = F.normalize(self.linear3(x))
        #print('7:', x.shape)
        return x.squeeze(1)
    

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, padding=2, dilation=1, stride=1):
        super().__init__()

        self.basic_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, padding=padding, dilation=dilation, stride=stride),
            nn.ReLU()
        )

    def forward(self, x):
        return self.basic_block(x)


class Classifier(nn.Module):
    def __init__(self,in_chanels=3):
        super().__init__()
        self.convBlock = nn.Sequential(
            Block(in_chanels, 8),
            nn.MaxPool1d(kernel_size=2),
            Block(8, 16),
            nn.MaxPool1d(kernel_size=2),
            Block(16, 32),
            nn.MaxPool1d(kernel_size=2),
            Block(32, 64),
            nn.MaxPool1d(kernel_size=2),
            Block(64, 64),
            nn.MaxPool1d(kernel_size=2),
            Block(64, 128),
            nn.MaxPool1d(kernel_size=2),
            Block(128, 256),
            nn.MaxPool1d(kernel_size=2),
            Block(256, 512)
        )
        self.classification = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.convBlock(x)
        x, _ = torch.max(x, dim=2)
        x = self.classification(x)
        return x
    

class Net5(nn.Module):
    def __init__(self,in_chanels=3):
        super().__init__()
        self.convBlock = nn.Sequential(
            Block(in_chanels, 8),
            nn.MaxPool1d(kernel_size=2),
            Block(8, 16),
            nn.MaxPool1d(kernel_size=2),
            Block(16, 32),
            nn.MaxPool1d(kernel_size=2),
            Block(32, 64),
            nn.MaxPool1d(kernel_size=2),
            Block(64, 64),
            nn.MaxPool1d(kernel_size=2),
            Block(64, 128),
            nn.MaxPool1d(kernel_size=2),
            Block(128, 256),
            nn.MaxPool1d(kernel_size=2),
            Block(256, 512)
        )
        self.classification = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.convBlock(x)
        #print('1:',x.size())
        x, _ = torch.max(x, dim=2)
        #print('2:',x.size())
        x = self.classification(x)
        #print('3:',x.size())
        return x
    