# Code
This folder will contain everything regarding the code for this project.

In the folder data you will find the data loader.

In the folder neural the architextures of used networks are saved.

In the folder output, the trained epochs and other outputs are stored.

In this folder you will find 2 jupyter notebooks. These we used for testing.

train_network.py is a script that trained a network. The networks architexture can be changed inside of the model section in the code. Whel ran, one needs to give a name to the output folder "output_name". Also here you can choose the number of epochs that the network is run and the trainset/ validationset sizes. Also the parameters of the optimizer can be changed here. The kind of criterion/optimizer can only be changed manually inside of the code.

test_network.py functions like train_network.py but only loops ofer the test set. I loads in an epoch of a given output folder (on default the last epoch). when ran, it prints the loss and accuracy of the network. It's a quick way to indicate if a network performs well.

data_visuallizaion.py is a script that combines the previous scrips. Without trainind it loops over every epoch and calculates the loss and accuracy of the 3 different data sets. It then stores this + the confusion matrix in a json txt file and saves it to the output folder. futhermore, it generates a loss and accuracy plot of all epochs wich are also saved to the output folder.
