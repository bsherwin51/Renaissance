from random import shuffle
import torch 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, os, time, h5py
from pathlib import Path

fluxSize = 200 #size of your wavelength array

def normalize_data(data, labels):
    ######################
    # normalize input
    ## the id's correspond to the id's of the feature array you made using vstack in read_data()
    data[:,0]  = (data[:,0] - np.mean(data[:,0]))/np.std(data[:,0])
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
def read_data(fin, seed, mode, normalize):

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

#getNormFactors function -- used in test.py
def getNormFactors(fin):
    f     = h5py.File(fin, 'r')
    returnarr = []
    try:
        redshift    = f['all_z'][:]
    except:
        redshift    = f['all_redshift'][:]
    
    mean_redshift = np.mean(redshift)
    std_redshift = np.std(redshift)
    returnarr.append(mean_redshift)
    returnarr.append(std_redshift)

    try:
        mean_sfr = np.mean(np.log10(f['all_sfr'][:]))
        std_sfr = np.std(np.log10(f['all_sfr'][:]))
    except:
        mean_sfr = np.mean(np.log10(f['all_sfr_10'][:]))
        std_sfr = np.std(np.log10(f['all_sfr_10'][:]))
    returnarr.append(mean_sfr)
    returnarr.append(std_sfr)
    try:
        mean_sfrsize = np.mean(np.log10(f['all_sfr_size'][:]))
        std_sfrsize = np.std(np.log10(f['all_sfr_size'][:]))

        mean_stsize = np.mean(f['all_stellar_size'][:])
        std_stsize = np.std(f['all_stellar_size'][:])

        returnarr.append(mean_sfrsize)
        returnarr.append(std_sfrsize)
        returnarr.append(mean_stsize)
        returnarr.append(std_stsize)
    except:
        pass
    
    mean_ms = np.mean(np.log10(f['all_stellar_mass'][:] + 1))
    std_ms = np.std(np.log10(f['all_stellar_mass'][:]+1))
    returnarr.append(mean_ms)
    returnarr.append(std_ms)

    mean_md = np.mean(np.log10(f['all_dust_mass'][:]+1))
    std_md = np.std(np.log10(f['all_dust_mass'][:]+1))
    returnarr.append(mean_md)
    returnarr.append(std_md)

    try:
        mean_mg = np.mean(np.log10(f['all_gas_mass'][:]+1))
        std_mg = np.std(np.log10(f['all_gas_mass'][:]+1))
        returnarr.append(mean_mg)
        returnarr.append(std_mg)
    except:
        pass

    mean_metal = np.mean(np.log10(f['all_metallicity'][:]))
    std_metal = np.std(np.log10(f['all_metallicity'][:]))
    returnarr.append(mean_metal)
    returnarr.append(std_metal)

    mean_metal_y = np.mean(np.log10(f['all_metallicity_young'][:]))
    std_metal_y = np.std(np.log10(f['all_metallicity_young'][:]))
    returnarr.append(mean_metal_y)
    returnarr.append(std_metal_y)
    returnarr.append(redshift)
    # returnArr = [mean_redshift, std_redshift, mean_sfr, std_sfr, mean_sfrsize, std_sfrsize, mean_ms, std_ms, mean_stsize, std_stsize, mean_md, std_md, mean_mg, std_mg, mean_metal, std_metal, mean_metal_y, std_metal_y, redshift]
    
    return returnarr

# sample of denormArray function -- test.py
def denormArrayTNG(fin, feature, featureValArray): #have not modified
    features = ["redshift", "SFR", "SFR_size", "StellSize", "M_star", "M_dust", "M_gas", "metallicity", "y_metallicity", 'sed']
    #get all normalization factors
    arrnorms = getNormFactors(fin)
    mean_redshift = arrnorms[0]
    std_redshift = arrnorms[1] 
    mean_sfr = arrnorms[2]
    std_sfr = arrnorms[3]
    # mean_sfrsize = arrnorms[4]
    # std_sfrsize = arrnorms[5]
    # mean_stsize = arrnorms[6]
    # std_stsize = arrnorms[7]
    mean_ms = arrnorms[4]
    std_ms = arrnorms[5]
    
    mean_md = arrnorms[6]
    std_md = arrnorms[7]
    # mean_mg = arrnorms[12]
    # std_mg = arrnorms[13]
    mean_metal = arrnorms[14]
    std_metal = arrnorms[15]
    mean_metal_y = arrnorms[16]
    std_metal_y = arrnorms[17]
    redshift = arrnorms[18]
    
    returnArr = []
    
    try:
        featHere = featureValArray.numpy()
    except AttributeError:
        featHere = featureValArray.cpu().numpy()
    
    if feature == 'M_star':
        for val in featHere:
            denorm1 = float(val)*std_ms + mean_ms
            denorm2 = 10**denorm1 
            returnArr.append(np.format_float_scientific(denorm2, precision=2))
    elif feature == 'M_gas':
        for val in featHere:
            denorm1 = float(val)*std_mg + mean_mg
            denorm2 = 10**denorm1
            returnArr.append(np.format_float_scientific(denorm2, precision=2))
    elif feature == 'M_dust':
        for val in featHere:
            denorm1 = float(val)*std_md + mean_md
            denorm2 = 10**denorm1
            returnArr.append(np.format_float_scientific(denorm2, precision=2))
    elif feature == 'redshift':
        for val in featHere:
            denorm = float(val)*std_redshift + mean_redshift
            returnArr.append(round(denorm, ndigits = 2))
    elif feature == 'SFR_size':
        for val in featHere:
            denorm = float(val)*std_sfrsize + mean_sfrsize
            returnArr.append(round(denorm, ndigits = 2))
    elif feature == 'StellSize':
        for val in featHere:
            denorm = float(val)*std_stsize + mean_stsize
            returnArr.append(round(denorm, ndigits = 2))
    elif feature == 'SFR':
        for val in featHere:
            denorm1 = float(val)*std_sfr + mean_sfr
            denorm2 = 10**denorm1
            returnArr.append(np.format_float_scientific(denorm2, precision = 1))
    elif feature == 'metallicity':
        for val in featHere:
            denorm = float(val)*std_metal + mean_metal
            denorm2 = 10**denorm - 1
            returnArr.append(np.format_float_scientific(denorm2, precision=2))
    elif feature == 'y_metallicity':
        for val in featHere:
            denorm = float(val)*std_metal_y + mean_metal_y
            denorm2 = 10**denorm - 1
            returnArr.append(np.format_float_scientific(denorm2, precision=2))
    else:
        print("Not a recognized feature name! Modify function or correct feature name")
        return None
        
    if len(returnArr) == 1:
        return returnArr[0]
    else:
        return returnArr

# denormSingle -- test.py
def denormSingle(fin, feature, featureVal, dataset):    
    # features = ["redshift", "SFR", "SFR_size", "StellSize", "M_star", "M_dust", "M_gas", "metallicity", "y_metallicity"]
    #get all normalization factors
    
    arrnorms = getNormFactors(fin)
    if 'tng' in dataset:
        mean_redshift = arrnorms[0]
        std_redshift = arrnorms[1] 
        mean_sfr = arrnorms[2]
        std_sfr = arrnorms[3]
        # mean_sfrsize = arrnorms[4]
        # std_sfrsize = arrnorms[5]
        # mean_stsize = arrnorms[6]
        # std_stsize = arrnorms[7]
        mean_ms = arrnorms[4]
        std_ms = arrnorms[5]
        
        mean_md = arrnorms[6]
        std_md = arrnorms[7]
        # mean_mg = arrnorms[12]
        # std_mg = arrnorms[13]
        mean_metal = arrnorms[8]
        std_metal = arrnorms[9]
        mean_metal_y = arrnorms[10]
        std_metal_y = arrnorms[11]
        redshift = arrnorms[12]
    else:
        mean_redshift = arrnorms[0]
        std_redshift = arrnorms[1] 
        mean_sfr = arrnorms[2]
        std_sfr = arrnorms[3]
        mean_ms = arrnorms[4]
        std_ms = arrnorms[5]
        mean_md = arrnorms[6]
        std_md = arrnorms[7]
        mean_mg = arrnorms[8]
        std_mg = arrnorms[9]
        mean_metal = arrnorms[10]
        std_metal = arrnorms[11]
        mean_metal_y = arrnorms[12]
        std_metal_y = arrnorms[13]
        redshift = arrnorms[14]
    
    val = featureVal.cpu().numpy()
    
    if feature == 'M_star':
        denorm1 = float(val)*std_ms + mean_ms
        denorm2 = 10**denorm1
        returnArr = (np.format_float_scientific(denorm2, precision=2))
    elif feature == 'M_gas':
        denorm1 = float(val)*std_mg + mean_mg
        denorm2 = 10**denorm1
        returnArr = (np.format_float_scientific(denorm2, precision=2))
    elif feature == 'M_dust':
        denorm1 = float(val)*std_md + mean_md
        denorm2 = 10**denorm1
        returnArr = (np.format_float_scientific(denorm2, precision=2))
    elif feature == 'redshift':
        denorm = float(val)*std_redshift + mean_redshift
        returnArr = (round(denorm, ndigits = 2))
    elif feature == 'SFR_size':
        denorm = float(val)*std_sfrsize + mean_sfrsize
        returnArr = (round(denorm, ndigits = 2))
    elif feature == 'StellSize':
        denorm = float(val)*std_stsize + mean_stsize
        returnArr = (round(denorm, ndigits = 2))
    elif feature == 'SFR':
        denorm1 = float(val)*std_sfr + mean_sfr
        denorm2 = 10**denorm1
        returnArr = (np.format_float_scientific(denorm2, precision=2))
    elif feature == 'metallicity':
        denorm1 = float(val)*std_metal + mean_metal
        denorm2 = 10**denorm1
        returnArr = (np.format_float_scientific(denorm2, precision=2))
    elif feature == 'y_metallicity':
        denorm1 = float(val)*std_metal_y + mean_metal_y
        denorm2 = 10**denorm1
        returnArr = (np.format_float_scientific(denorm2, precision=2))
    else:
        print("Not a recognized feature name! Modify function or correct feature name")
        return None
    
    return returnArr

#Calculate fractional difference of obs and exp & fractional difference of std of exp
def calcFDiff(pred, exp, logged=True):
    totfracdiff = []
    stdfracdiff = []

#if you're not converting your inputs to numpy before plugging into the function, then uncomment this
#     try:
#         exp = exp.numpy()
#     except:
#         exp = exp.cpu().numpy()
        
    loggedstdExp = float((exp).std())
    unloggedstdExp = float(np.log10(exp.std()))
    
    for q, val in enumerate(pred):
        if logged:
            fdiff = float(val - exp[q])
            
            stddiff = val - exp[q]
            stdfdiff = stddiff/loggedstdExp
        else:
            fdiff = float(np.log10(val) - np.log10(exp[q]))
            
            stddiff = np.log10(val) - np.log10(exp[q])
            stdfdiff = stddiff/unloggedstdExp
        
        try:
            fdiff = np.asarray(fdiff)
            stdfdiff = np.asarray(stdfdiff)
        except Exception as e:
            print(e)
        
        # given an expected and true SED, outputs the fractional differences per wavelength in an array
        totfracdiff.append(fdiff)
        stdfracdiff.append(stdfdiff)
    
    toreturn = {'fracdiff': np.asarray(totfracdiff), 'fracdiffstd': np.asarray(stdfracdiff)}
    
    return toreturn