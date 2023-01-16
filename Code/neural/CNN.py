import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """ Generates a CNN """
    def __init__(self,num_channels: int, num_kernels: int, kernel_dim:int):
        """"""
        super(CNN, self).__init__()
        self.num_channels = num_channels

        output_dim = num_kernels * num_channels
        self.conv1 = nn.Conv1d(num_channels, output_dim, kernel_size=(kernel_dim,), bias=False, groups=num_channels)
        
        self.conv2 = nn.Conv1d(num_channels, output_dim, kernel_size=(kernel_dim,), bias=False, groups=num_channels,dilation=4)
        
       
        self.linear1 = nn.Linear(65436, 256)
        self.linear2 = nn.Linear(256, 32)
        self.linear3 = nn.Linear(32,2)

    def add(self, layer):
        return layer.sum(1).type(torch.float32)

    def set_kernel(self,kernels):
        with torch.no_grad():
            #kernels = torch.cat(kernels, dim=1)
            #kernels = kernel.conj().T.unsqueeze(1)

            #len_diff = kernels.shape[-1] - self.conv1.weight.shape[-1]
            #kernels = kernels[:, :, len_diff//2:(kernels.shape[-1] - len_diff//2)]
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


