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
        x = F.normalize(self.linear2(x))
        #print(x)
        #print('5:', x.shape)
        return x


