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
parser.add_argument("--weight_decay", type=float, default=0)
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
    model = Net5()
    model = model.to(device)
    
   
    #-#load checkpoint
    model_checkpoints = [str(f) for f in Path(f'{output_path}/').glob("checkpoint_*")]
    sorted_model_checkpoint = sorted(model_checkpoints, key=lambda x:float(x.split(f"checkpoint_epoch_")[1].split("_loss")[0]))
    last_checkpoint = torch.load(sorted_model_checkpoint[-1])
    training_args = last_checkpoint['args']
    
    #Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    
    #handles the distribution of train/validation/test sets
    data_distribution = len(data)*np.array(training_args.distribution_trn_val_tst) 
    bounds = np.floor(np.cumsum(data_distribution))

    #generates a testset with all testsamples
    testset_injections = data.get_injections(np.arange(bounds[1],bounds[2],dtype=int))
    testset_backgrounds = data.get_backgrounds(np.arange(bounds[1],bounds[2],dtype=int))
    
    test_labels= np.concatenate((np.repeat([[0,1]],len(testset_injections),axis=0),
                                 np.repeat([[1,0]],len(testset_backgrounds),axis=0)))
    test_data = np.concatenate((testset_injections,testset_backgrounds),0)
    
    testset = (torch.tensor(test_data), torch.tensor(test_labels))
    Testset = Data_set_transform(testset)
    
    testloader = torch.utils.data.DataLoader(Testset, batch_size=training_args.batch_size,
                                              shuffle=True, num_workers=training_args.num_workers)
    print('all test data loaded')
    
    #Load data in memory
    trainset = data.make_dataset(training_args.trainsize, data_bounds=[0,bounds[0]])
    Trainset = Data_set_transform(trainset)

    valset = data.make_dataset(training_args.valsize, data_bounds=[bounds[0],bounds[1]])
    Valset = Data_set_transform(valset)

    trainloader = torch.utils.data.DataLoader(Trainset, batch_size=training_args.batch_size,
                                              shuffle=True, num_workers=training_args.num_workers)

    valloader = torch.utils.data.DataLoader(Valset, batch_size=training_args.batch_size,
                                            shuffle=True, num_workers=training_args.num_workers)
    print('train and validation data loaded')
    
    #init arrays to store and save the data
    epochs_train_accuracy = []
    epochs_train_loss = []
    epochs_val_accuracy = []
    epochs_val_loss = []
    epochs_test_accuracy = []
    epochs_cunfusion_matrix = []
    epochs_test_loss = []
    for epoch in range(len(sorted_model_checkpoint)):
        checkpoint = torch.load(sorted_model_checkpoint[epoch])
        
        # Set the correct values
        loaded_epoch = checkpoint['epoch'] 
        print(f'loaded epoch:{loaded_epoch}')
        
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        
        #loop without learing
        with torch.set_grad_enabled(False):
            #init parameters
            train_accuracy = 0
            running_train_loss = 0

            val_accuracy = 0
            running_val_loss = 0

            test_accuracy = 0
            cunfusion_matrix = np.zeros((2,2))
            running_test_loss = 0

            print('Starting training set:')
            for batch, local in enumerate(trainloader):
                # Transfer to GPU
                local_batch, local_labels = local
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Model computations
                outputs = model(local_batch.type(torch.float32))#.unsqueeze(1)

                #loss in batch
                train_loss = criterion(outputs, local_labels.type(torch.float32))
                running_train_loss += train_loss.item()

                #tests if the network get the correct label to take the highest percentage      
                for Label, Output in zip(local_labels, outputs):
                    label = Label.clone().detach().cpu().numpy() 
                    output = Output.clone().squeeze(0).detach().cpu().numpy()
                    if label[0]>label[1] and output[0]>output[1]:
                        #Correct classification as background
                        train_accuracy +=1
                    if label[0]<label[1] and output[0]<output[1]:
                        #Correct classification as injection
                        train_accuracy +=1

            #avarage loss
            avarage_train_loss = running_train_loss / (2*training_args.trainsize/training_args.batch_size)
            print(f'training loss: {avarage_train_loss:.3f}')
            epochs_train_loss.append(avarage_train_loss)

            #total accuracy
            total_train_accuracy = train_accuracy/(2*training_args.trainsize)*100
            print(f'training accuracy:{total_train_accuracy:.2f}%')
            epochs_train_accuracy.append(total_train_accuracy)
            
            
            print('Starting validation set:')
            for batch, local in enumerate(valloader):
                # Transfer to GPU
                local_batch, local_labels = local
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Model computations
                outputs = model(local_batch.type(torch.float32))#.unsqueeze(1)
                
                #loss in batch
                val_loss = criterion(outputs, local_labels.type(torch.float32))
                running_val_loss += val_loss.item()

                #tests if the network get the correct label to take the highest percentage      
                for Label, Output in zip(local_labels, outputs):
                    label = Label.clone().detach().cpu().numpy() 
                    output = Output.clone().squeeze(0).detach().cpu().numpy()
                    if label[0]>label[1] and output[0]>output[1]:
                        #Correct classification as background
                        val_accuracy +=1
                    if label[0]<label[1] and output[0]<output[1]:
                        #Correct classification as injection
                        val_accuracy +=1

            #avarage loss
            avarage_val_loss = running_val_loss / (2*training_args.valsize/training_args.batch_size)
            print(f'validation loss: {avarage_val_loss:.3f}')
            epochs_val_loss.append(avarage_val_loss)

            #total accuracy
            total_val_accuracy= val_accuracy/(2*training_args.valsize)*100
            print(f'validation accuracy:{total_val_accuracy:.2f}%')
            epochs_val_accuracy.append(total_val_accuracy)
            
            
                        
            print('Starting test set:')
            for batch, local in enumerate(testloader):
                # Transfer to GPU
                local_batch, local_labels = local
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Model computations
                outputs = model(local_batch.type(torch.float32))#.unsqueeze(1)

                #loss in batch
                test_loss = criterion(outputs, local_labels.type(torch.float32))
                running_test_loss += test_loss.item()

                #tests if the network get the correct label to take the highest percentage      
                for Label, Output in zip(local_labels, outputs):
                    label = Label.clone().detach().cpu().numpy() 
                    output = Output.clone().squeeze(0).detach().cpu().numpy()
                    if label[0]>label[1] and output[0]>output[1]:
                        #Correct classification as background
                        test_accuracy +=1
                        cunfusion_matrix[0,0] +=1
                    if label[0]<label[1] and output[0]<output[1]:
                        #Correct classification as injection
                        test_accuracy +=1
                        cunfusion_matrix[1,1] +=1
                    if label[0]>label[1] and output[0]<output[1]:
                        #Incorrect classification as  injection, but was background
                        cunfusion_matrix[1,0] +=1
                    if label[0]<label[1] and output[0]>output[1]:
                        #Incorrect classification as background, but was injection
                        cunfusion_matrix[0,1] +=1


            #avarage loss
            avarage_test_loss = running_test_loss / (len(test_data)/training_args.batch_size)
            print(f'test loss: {avarage_test_loss:.3f}')
            epochs_test_loss.append(avarage_test_loss)

            #total accuracy
            total_test_accuracy= test_accuracy/(len(test_data))*100
            print(f'test accuracy:{total_test_accuracy:.2f}%')
            epochs_cunfusion_matrix.append(cunfusion_matrix)
            epochs_test_accuracy.append(total_test_accuracy)
            
    plt.figure()
    plt.plot(epochs_train_loss,label='avarage training loss')
    plt.plot(epochs_val_loss, label= 'avarage validation loss')
    plt.plot(epochs_test_loss, '--', label= 'avarage test loss')
    plt.legend()
    plt.grid()
    plt.ylabel('Loss (CrossEntropy)')
    plt.xlabel('epoch')
    plt.savefig(f'{output_path}/Losses_plot.pdf')
    plt.close()
    
    plt.figure()
    plt.plot(epochs_train_accuracy, '--',label='training set')
    plt.plot(epochs_val_accuracy, '--', label= 'validation set')
    plt.plot(epochs_test_accuracy, label= 'test set')
    plt.legend()
    plt.grid()
    plt.ylabel('Accuracy (%)')
    plt.xlabel('epoch')
    plt.savefig(f'{output_path}/Accuracy_plot.pdf')
    plt.close()
    
    print(epochs_cunfusion_matrix[0])
    with open(f'{output_path}/data_visualization_arrays.txt', 'w') as f1:
            json.dump({'epochs_train_loss':epochs_train_loss, 'epochs_train_accuracy':epochs_train_accuracy,
                       'epochs_val_loss':epochs_val_loss, 'epochs_val_accuracy':epochs_val_accuracy,
                       'epochs_test_loss':epochs_test_loss, 'epochs_test_accuracy':epochs_test_accuracy,
                       'epochs_cunfusion_matrix':np.array(epochs_cunfusion_matrix).tolist()}, f1, indent=4)