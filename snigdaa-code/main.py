import numpy as np
import sys, os, time
import torch 
import torch.nn as nn
from pathlib import Path

import data, architecture
# data parameters
##############
fin  = '/mnt/home/ssethuram/ceph/datasets/tng50_SHUFFLEDDATA.h5' # input data file
dataname = 'TNG'        # nickname for your data
seed = 20                # remember this to reproduce split
numFeatures = 5         # number of input properties
fluxSize = 200          # size of wavelength array
featurelist = 'SFR, M_dust, M_star, metal, metaly'
# architecture parameters
numHL = 1            # number of hidden layers
# h1 = 954                # nodes in first hidden layer
h1 = 419              # nodes in second hidden layer

# dr1 = 0.20103              # dropout rate for layer 1
dr1 = 0.4544
# training parameters
batch_size = 128
# lr         = 1.404e-3
lr = 0.00389
epochs     = 1500
# wd         = 1.195e-6
wd = 1.083e-5
##############

# name of output files
name   = "tng_dynamicmodel2"

# NOT SURE WHAT TO DO HERE TO LINE 60
try: 
    Path('./losses').mkdir(parents=False, exist_ok=False)
    Path('./models').mkdir(parents=False, exist_ok=False)
except Exception as e:
    print(e)

fout   = 'losses/%s.txt'%name
fmodel = 'models/%s.pt'%name

#write the header
f = open(fout, 'a')
f.write('Training dataset: {}\n'.format(dataname))
f.write('seed = {}\nnumFeatures = {}\nfluxSize = {}\nh1 = {}\ndr1 = {}\nbatch_size = {}\nlr = {}\nepochs = {}\nwd = {}\n'.format(seed, numFeatures, fluxSize, h1,dr1, batch_size, lr, epochs, wd))
f.write('0 epoch  1 train loss  2 valid loss  3 test loss\n')
f.close()
#done writing header

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# define loss function
criterion = nn.MSELoss() 

# get train, validation, and test sets
print('preparing dataset...')
train_loader = data.create_dataset('train', seed, fin, batch_size, shuffle=True)
valid_loader = data.create_dataset('valid', seed, fin, batch_size, shuffle=False)
test_loader  = data.create_dataset('test',  seed, fin, batch_size, shuffle=False)

#########################
# define architecture
# model = architecture.model_1hl(numFeatures, h1, fluxSize, dr1) # change function according to number of hidden layers
model = architecture.dynamic_model2(numFeatures, fluxSize, numHL, [h1], [dr1])
#########################

#casts parameters/buffers to specified gpu
model.to(device=device)
#get num elements in each network parameters and sum them
network_total_params = sum(p.numel() for p in model.parameters())
print('total number of parameters in the model = %d'%network_total_params)

# define optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.999), 
                             weight_decay=wd)	

# load best-model, if it exists
if os.path.exists(fmodel):  
    print('Loading model...')
    model.load_state_dict(torch.load(fmodel))

# get validation loss
print('Computing initial validation loss')
model.eval()
min_valid_loss, points = 0.0, 0
for x, y in valid_loader:
    with torch.no_grad():
        x    = x.to(device=device)
        y    = y.to(device=device)
        y_NN = model(x)
        min_valid_loss += (criterion(y_NN, y).item())*x.shape[0]
        points += x.shape[0]
min_valid_loss /= points
print('Initial valid loss = %.3e'%min_valid_loss)

# see if results for this model are available (if you're continuing training on a model)
if os.path.exists(fout):  
    dumb = np.loadtxt(fout, skiprows = 11, unpack=False)
    if dumb.size == 0:
        offset = 0
    else: offset = int(dumb[:,0][-1]+1)
else:   offset = 0

# do a loop over all epochs
start = time.time()
for epoch in range(offset, offset+epochs):
    # do training
    train_loss, points = 0.0, 0
    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        y_NN = model(x)

        loss = criterion(y_NN, y)
        train_loss += (loss.item())*x.shape[0]
        points     += x.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= points

    # do validation
    valid_loss, points = 0.0, 0
    model.eval()
    for x, y in valid_loader:
        with torch.no_grad():
            x    = x.to(device)
            y    = y.to(device)
            y_NN = model(x)
            valid_loss += (criterion(y_NN, y).item())*x.shape[0]
            points     += x.shape[0]
    valid_loss /= points

    # do testing
    test_loss, points = 0.0, 0
    model.eval()
    for x, y in test_loader:
        with torch.no_grad():
            x    = x.to(device)
            y    = y.to(device)
            y_NN = model(x)
            test_loss += (criterion(y_NN, y).item())*x.shape[0]
            points    += x.shape[0]
    test_loss /= points

    # save model if it is better
    if valid_loss<min_valid_loss:
        torch.save(model.state_dict(), fmodel)
        min_valid_loss = valid_loss
        print('%03d %.3e %.3e %.3e (saving)'%(epoch, train_loss, valid_loss, test_loss))
    else:
        print('%03d %.3e %.3e %.3e'%(epoch, train_loss, valid_loss, test_loss))
    
    # save losses to file
    f = open(fout, 'a')
    f.write('%d %.5e %.5e %.5e\n'%(epoch, train_loss, valid_loss, test_loss))
    f.close()
    
stop = time.time()
print('Time take (m):', "{:.4f}".format((stop-start)/60.0))
