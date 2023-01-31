#Data loader

In this work, we use a generated data-set, which resembles how observations of cosmic strings would appear in three different gravitational wave detectors. 
An observation event is a one-dimensional time series of 8 seconds at a sampling frequency of 8096 Hz, which makes an array of 65536 values. 
And each batch of data has three such time series, representing different detectors. In total, there are 30,000 of these triplets. 
Half of the data set consists of just Gaussian noise, simulating observations when there are no signals from cosmic strings. 
These are called as ’Background signals’ and are exemplified by Fig. 2a. 
The other class of data contains wave forms of cosmic string signals – which were generated according to their theoretical description – 
in addition to the Gaussian noise. This class of data is called ’Injection signals’.
_Data was profided by Malissa Lopes on the Nikhef cluster_

In this folder you will find a dataloader. The Data-loader is a code that loads the data available from the data set to our network, it keeps the different
types of data distinct accordingly to the two labels: ’Injection’ or ’Background’. Specificcaly the _generate_dataset()_ function. It also shuffles the order of the data, to remove any potential biases in the way the data
was ordered. 

When loaded in memory for traing/testing, the data is separated into batches of size 32, which improves the computational efficiency of the network. 
The data is further divided into three sets. "Training data" is used by the network to learn and optimize its parameters. 
In our case, we separated the data according to 80% as training data, 10% as validation data, and 10% as test data. This is partitioned in the order the data is saved.

While training, during every epoch, 4,096 (2,048 injectionsand 2,048 backgrounds) training data samples, and 1024
(512 injections, 512 backgrounds) validation samples were drawn from each of these groups and for each epoch.
