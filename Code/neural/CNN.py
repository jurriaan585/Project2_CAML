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
        self.conv1 = nn.Conv1d(num_channels, output_dim, kernel_size=(kernel_dim,), bias=False, groups=3)

        self.linear1 = nn.Linear(output_dim, 256)
        self.linear2 = nn.Linear(256, 32)
        self.linear3 = nn.Linear(32,1)

    def add(self, layer):
        return layer.sum(0)

    def set_kernel(self,kernels):
        with torch.no_grad():
            #kernels = torch.cat(kernels, dim=1)
            #kernels = kernel.conj().T.unsqueeze(1)

            #len_diff = kernels.shape[-1] - self.conv1.weight.shape[-1]
            #kernels = kernels[:, :, len_diff//2:(kernels.shape[-1] - len_diff//2)]
            self.conv1.weight = torch.nn.Parameter(kernels)

    def forward(self, x):
        with torch.no_grad():
            x = self.conv1(x)
        x = self.add(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


