from random import shuffle
import torch 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, os, time, h5py
from pathlib import Path

def normalize_data(data, labels):
    ######################
    # normalize input
    ## the id's correspond to the id's of the feature array you made using vstack in read_data()
    data[:,0]  = (data[:,0] - np.mean(data[:,0]))/np.std(data[:,0]) # Z scores
    data[:,1]  = (data[:,1] - np.mean(data[:,1]))/np.std(data[:,1])
    data[:,2]  = (data[:,2] - np.mean(data[:,2]))/np.std(data[:,2])
    data[:,3]  = (data[:,3] - np.mean(data[:,3]))/np.std(data[:,3])
    data[:,4]  = (data[:,4] - np.mean(data[:,4]))/np.std(data[:,4])

    ######################
    # normalize labels
    print ("labels", labels)

    array = np.log10(labels)
    print ("labels array", array)
    labels = array
    
    return data, labels

# read data and get training, validation or testing sets
# fin ---------> file with the data
# seed --------> random seed used to split among different datasets
# mode --------> 'train', 'valid', 'test' or 'all'
# normalize ---> whether to normalize the data or not
def read_data(fin, seed, mode, normalize): #fin for h5

    fluxFile = '/mnt/home/ssethuram/ceph/datasets/tngFluxesOriginal_shuffled.txt'
    ######################
    # read data - EDIT
    f     = h5py.File(fin, 'r')
    redshift    = f['all_z'][:]
    SFR    = f['all_sfr'][:]
    M_dust     = f['all_dust_mass'][:]
    # fluxes = f['all_fluxes'][:]
    # M_gas  = f['all_gas_mass'][:]
    M_star  = f['all_stellar_mass'][:]  
    metal = f['all_metallicity'][:]  
    metaly = f['all_metallicity_young'][:]  
    f.close()
    
    ######################
    # normalize data - EDIT
    ## anything not on order 10^0 - 10^1 should be logged
    SFR = np.log10(SFR)
    M_dust = np.log10(M_dust)
    # M_gas = np.log10(M_gas)
    M_star = np.log10(M_star)
    metal = np.log10(metal)
    metaly = np.log10(metaly)

    # get data, labels and number of elements
    data = np.vstack([SFR, M_dust, M_star, metal, metaly]).T # THESE ARE YOUR CHOSEN INPUT VARIABLES
    
    labels = np.loadtxt(fluxFile)
    print ("shape of fluxes", labels.shape)
    # labels = fluxes.reshape((fluxes.shape[0], fluxSize))
    elements = data.shape[0]

    # normalize data
    if normalize:  data, labels = normalize_data(data, labels)

    # get the size and offset depending on the type of dataset
    if   mode=='train':   size, offset = int(elements*0.70), int(elements*0.00)
    elif mode=='valid':   size, offset = int(elements*0.15), int(elements*0.70)
    elif mode=='test':    size, offset = int(elements*0.15), int(elements*0.85)
    elif mode=='all':     size, offset = int(elements*1.00), int(elements*0.00)
    else:                 raise Exception('Wrong name!')

    # randomly shuffle the cubes. Instead of 0 1 2 3...999 have a 
    # random permutation. E.g. 5 9 0 29...342
    np.random.seed(seed)
    indexes = np.arange(elements) 
    np.random.shuffle(indexes)
    indexes = indexes[offset:offset+size] #select indexes of mode

    return data[indexes], labels[indexes]


# This class creates the dataset 
class make_dataset():

    def __init__(self, mode, seed, fin):

        # get data
        inp, out = read_data(fin, seed, mode, normalize=True)

        # get the corresponding bottlenecks and parameters
        self.size   = inp.shape[0]
        self.input  = torch.tensor(inp, dtype=torch.float32)
        self.output = torch.tensor(out, dtype=torch.float32)
        
        print ("size of input and output", np.shape(self.input), np.shape(self.output))
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]


# This routine creates a dataset loader
def create_dataset(mode, seed, fin, batch_size, shuffle=False):
    data_set = make_dataset(mode, seed, fin)
    dataset_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle)
    return dataset_loader

