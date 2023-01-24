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

from neural.Network import *

#-# Make arguments parser
parser = argparse.ArgumentParser("parameters for Neural network, Cosmic strings gravitational waves")

#Name the output folder
parser.add_argument("--output_name", type=str, default='temp')

#Data arguments
parser.add_argument("--distribution_trn_val_tst", type=float, nargs='+', default= [0.8,0.1,0.1]) 
parser.add_argument("--testsize", type=int, default=int(2048))

parser.add_argument("--epoch", type=int, default=int(-1))

parser.add_argument("--batch_size", type=int, default=int(32))
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
    
    args = parser.parse_args()
    
    assert sum(args.distribution_trn_val_tst) == 1, 'The give data distibution does not add up to 1'
    
    args.output_name = input('Type here the name of the output: ') if args.output_name == 'temp' else args.output_name
    args.output_name = ''.join([i if i != ' ' else '_' for i in args.output_name])
    args.output_name = 'temp' if args.output_name == '' else args.output_name
    output_path = f'output/{args.output_name}'
    # Create directory or load the previous session
    if not Path(output_path).exists():
        Exception('ERROR: Incorrect output path given')
        
    
    path = '/data/gravwav/lopezm/ML_projects/Projects_2022/Project_2/Data_new'
        
    #create testset
    data = GW_DS_NEW(path)
    
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
    model = Net4()#Net3(data.array_length,len(normalized_waveform))#Net2()
    kernel = kernels[0].squeeze(0).unsqueeze(1)
    #print(kernel.size())
    #model.set_kernel(kernel)#model.set_kernels(kernels)
    model = model.to(device)
    
    #Optimizer
    criterion = nn.MSELoss()#nn.CrossEntropyLoss()#
    #optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)#, momentum=args.momentu
    
    
    #-#load checkpoint
    model_checkpoints = [str(f) for f in Path(f'{output_path}/').glob("checkpoint_*")]
    sorted_model_checkpoint = sorted(model_checkpoints, key=lambda x:float(x.split(f"checkpoint_epoch_")[1].split("_loss")[0]))
    checkpoint = torch.load(sorted_model_checkpoint[args.epoch])

    # Set the correct values
    loaded_epoch = checkpoint['epoch'] 
    print(f'loaded epoch:{loaded_epoch}')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    
    #handles the distribution of train/validation/test sets
    data_distribution = len(data)*np.array(args.distribution_trn_val_tst) 
    bounds = np.floor(np.cumsum(data_distribution))
    
    
    testset = data.make_dataset(args.testsize, data_bounds=[bounds[1],bounds[2]])
    Testset = Data_set_transform(testset)
    testloader = torch.utils.data.DataLoader(Testset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)
    print('test data loaded')
    
    Correctness = 0 #initinalizing how many samples were correct 
    running_test_loss = 0
    with torch.set_grad_enabled(False):
        for batch, local in enumerate(testloader):
            print(f'batch:{batch} starting')
            # Transfer to GPU
            local_batch, local_labels = local
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            outputs = model(local_batch.type(torch.float32))#.unsqueeze(1)

            #loss in batch
            test_loss = criterion(outputs, local_labels.type(torch.float32))
            print(f'batch:{batch} loss:{test_loss:.2f}')
            running_test_loss += test_loss.item()
            
            #test if the network get the correct label to take the highest percentage
            local_Correctness = 0
            for truth, guess in zip(local_labels, outputs):
                Truth = truth.clone().detach().cpu().numpy() 
                Guess = guess.clone().squeeze(0).detach().cpu().numpy()
                if Truth[0]>Truth[1] and Guess[0]>Guess[1] or Truth[0]<Truth[1] and Guess[0]<Guess[1]:
                    Correctness += 1 
                    local_Correctness += 1
            print(f'batch:{batch} Correctness:{local_Correctness:.0f} out of {args.batch_size:.0f}, accuracy:{local_Correctness/args.batch_size*100:.2f}%')
                

        #avarage loss
        avarage_test_loss = running_test_loss / (2*args.testsize/args.batch_size)
        print(f'test loss: {avarage_test_loss:.3f}')
        
        #total correctness and accuracy
        print(f'Correctness:{Correctness:.0f} out of {2*args.testsize:.0f}, accuracy:{Correctness/(2*args.testsize)*100:.2f}%')
    
    """
    sample = Testset[0]
    
    signal, label = sample
    fig, ax = plt.subplots(3,1)
    for i in [1,2,3]:
        ax[i-1].plot(signal[i-1], label=f'E{i}')
        ax[i-1].legend()
    plt.savefig(output_path+'/sample.pdf')

    Prediction = model(signal.unsqueeze(0).unsqueeze(1).type(torch.float32))
    print(label, Prediction[0][0])
    Label = 'Signal' if label[0] == 1 else 'Background'
    print(f'the network guesses sample to be {Prediction[0,0,0]**2*100:.2f}%Signal and {Prediction[0,0,1]**2*100:.2f}%Background, \n The label was {Label}.')
    """