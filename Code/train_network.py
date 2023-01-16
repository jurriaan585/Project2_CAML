import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data.Data_loader import GravitationalWave_datastrain as GW_DS, GravitationalWave_datastrain_New as GW_DS_NEW
from data.Data_loader import Data_set_transform 

from neural.Network import Net as Net

#-# Make arguments parser
parser = argparse.ArgumentParser("parameters for Neural network, Cosmic strings gravitational waves")

#Name the output folder
parser.add_argument("--output_name", type=str, default='temp')

#Data arguments
parser.add_argument("--distribution_trn_val_tst", type=float, nargs='+', default= [0.8,0.1,0.1]) 
parser.add_argument("--trainsize", type=int, default=int(2048))
parser.add_argument("--valsize", type=int, default=int(512))
parser.add_argument("--testsize", type=int, default=int(512))

parser.add_argument("--num_epochs", type=int, default=int(10))
parser.add_argument("--batch_size", type=int, default=int(16))
parser.add_argument("--num_workers", type=int, default=2)
#parser.add_argument("--optimizer", type=str, default="AdamW", choices=["SGD", "Adam", "AdamW"])
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--momentum", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--device", type=str, default="cuda:0", choices=["cuda:0", "cpu:0"])


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device(0)
        print("GPU OK")
    else:
        device = "cpu"
        print("no GPU")
        
    #Get parameters
    args = parser.parse_args()
    
    assert sum(args.distribution_trn_val_tst) == 1, 'The give data distibution does not add up to 1'
    
    args.output_name = input('Type here the name of the output: ') if args.output_name == 'temp' else args.output_name
    args.output_name = ''.join([i if i != ' ' else '_' for i in args.output_name])
    args.output_name = 'temp' if args.output_name == '' else args.output_name
    output_path = f'output/{args.output_name}'
    # Create directory or load the previous session
    if not Path(output_path).exists():
        Path(output_path).mkdir(exist_ok=True, parents=True)
    
       
    #Make the data loader objects
    path = '/data/gravwav/lopezm/ML_projects/Projects_2022/Project_2/Data_new'
    data_old = GW_DS(path[:-4]) #remnent for if needed
    data = GW_DS_NEW(path)
    
    #handles the distribution of train/validation/test sets
    data_distribution = len(data)*np.array(args.distribution_trn_val_tst) 
    bounds = np.floor(np.cumsum(data_distribution))


    #-# Model of the network
    #get kernal
    waveform = np.load(path+f'/Waveforms/wf{0}.npy.html').T[1]
    normalized_waveform = waveform / np.linalg.norm(waveform)
    kernels = np.zeros((8,3,len(normalized_waveform))) 
    for kernel_indx in range(len(kernels)):
        for waveform_indx, flip_waveform in enumerate(f'{kernel_indx:0{len(kernels[0])}b}'):
            kernels[kernel_indx][waveform_indx] = (-1)**int(flip_waveform) * normalized_waveform
    kernels = torch.tensor(kernels).type(torch.float32) 
    
    #model
    model = Net()
    model.set_kernels(kernels)
    

    #Optimizer
    criterion = nn.MSELoss()#CrossEntropyLoss()#
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    
    #-# Loop over epochs
    for epoch in range(args.num_epochs):
        #Load data in memory
        print(f'epoch:{epoch} loading data')
        trainset = data.make_dataset(args.trainsize, data_bounds=[0,bounds[0]])
        Trainset = Data_set_transform(trainset)
        
        valset = data.make_dataset(args.valsize, data_bounds=[bounds[0],bounds[1]])
        Valset = Data_set_transform(valset)

        #Not used yet
        #testset = data.make_dataset(args.testsize, data_bounds=[bounds[1],bounds[2]])

        trainloader = torch.utils.data.DataLoader(Trainset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=args.num_workers)

        valloader = torch.utils.data.DataLoader(Valset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.num_workers)

        #Not used yet
        #testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
        #                                          shuffle=True, num_workers=args.num_workers)
        
        print(f'epoch:{epoch} data loaded')
        
        # Training
        running_train_loss = 0
        for batch, local in enumerate(trainloader):
            
            print(f'epoch:{epoch} training batch:{batch} starting')
            # Transfer to GPU
            local_batch, local_labels = local
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            outputs = model(local_batch.unsqueeze(1).type(torch.float32))
        
            #loss in batch
            # zero the parameter gradients
            optimizer.zero_grad()
            train_loss = criterion(outputs, local_labels.type(torch.float32))
            train_loss.backward()
            print(f'epoch:{epoch} training batch:{batch} loss:{train_loss:.2f}')
            running_train_loss += train_loss.item()
            
            #optimize
            optimizer.step()
            
        avarage_train_loss = running_train_loss / (2*args.trainsize/args.batch_size)
        print(f'epoch:{epoch}, train loss: {avarage_train_loss:.3f}')
        
        # Validation
        running_val_loss = 0
        with torch.set_grad_enabled(False):
            for index, local in enumerate(valloader):
                
                print(f'epoch:{epoch} validation batch:{batch} starting')
                # Transfer to GPU
                local_batch, local_labels = local
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                
                # Model computations
                outputs = model(local_batch.unsqueeze(1).type(torch.float32))
                
                #loss in batch
                val_loss = criterion(outputs, local_labels.type(torch.float32))
                print(f'epoch:{epoch} training batch:{batch} loss:{val_loss:.2f}')
                running_val_loss += val_loss.item()
                                
            #avarage loss
            avarage_val_loss = running_val_loss / (2*args.valsize/args.batch_size)
            print(f'epoch:{epoch + 1}, validation loss: {avarage_val_loss:.3f}')


        # Save model
        torch.save(obj={'epoch': epoch,'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       f=f'{output_path}/checkpoint_epoch_{epoch}_loss_{running_val_loss:.3f}.pth')