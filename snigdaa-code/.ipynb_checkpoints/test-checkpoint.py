# test out your optuna neural network
# based on a script in Francisco Villaescusa-Navarro's Pylians3_ML repository

## this code was written very much for convenience and not necessarily for efficiency, so if you have input 
## on any of these scripts, please submit a pull request!
## if I find some time in the future with nothing to do, I may come back and 
## improve/eliminate some of these loops and redundancies

from unicodedata import name
import numpy as np
import sys, os, time, h5py
from pathlib import Path
import torch
import torch.nn as nn
import data, architecture
import optuna
from statistics import median

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import time
import matplotlib.cm as cm

#user-defined function
def sci_notation(feature, unit, number, sig_fig=2):
    # ret_string = "{0:.{1:d}s}".format(number, sig_fig)
    try:
        a = number[:number.index('e')]
        b = int(number[number.index('+'):])
    except:
        a = number[:number.index('e')]
        b = int(number[number.index('-'):])
    if b == 0:
        return "%s / %s : %s" % (feature, unit, a)
        # print(feature)
        # print(number)
        # print(type(number))
    # print(ret_string)
    # print(ret_string.split("e"))
    # try:
    # a, b = ret_string.split("e")
    # except:
    #     a = ret_string.split("e")
    #     b = 0
    # # remove leading "+" and strip leading zeros
    # b = int(b)
    else:
        return "%s / %s : %s x 10$^{%s}$" % (feature, unit, a, b)

# print(sci_notation('SFR', '[$M_{\odot}$ $yr^{-1}$]', np.format_float_scientific(123456788, precision=2), sig_fig=4))
# STOP

################################### INPUT ############################################
# data parameters
f_features      = '/mnt/home/ssethuram/ceph/datasets/tngFeatures_shuffled.npy' # path to your input features file (global properties). How the data is read is dependent on your create_dataset function in data.py. Modify as needed   
f_features_norm = None 
f_labels  = '/mnt/home/ssethuram/ceph/datasets/tngFluxesOriginal_shuffled.txt' # path to your output labels file (training SEDs). Check data.py for details on normalization
seed      = 20   # seed to split data in train/valid/test; set to any value, remember it if you want to reproduce the split
mode      = 'test'   # 'train','valid','test' or 'all' -- check data.py for what this means
  
# optuna/architecture parameters
## enter the same parameters you used in main.py to run Optuna! Necessary to find the right database
input_size  = 5   # dimensions of input data (number of input global properties)
numHL = 1 # maximum number of hidden layers to test
hidden = [419]
dr = [0.4544]
batch_size = 128 # batch size for training model
epochs     = 1500 # num epochs for training model

# optuna & file parameters
fin = '/mnt/home/ssethuram/ceph/datasets/tng50_SHUFFLEDDATA.h5' #this is the file you will use for normalization (enter the data file you used to train in main.py. This will be especially important to specify if cross-testing datasets)
fname = 'tng_dynamicmodel2' #enter the study name you set in main.py
output_size = 200 #this is the size of your wavelength array

## set name -- should be the same as in main.py
wavelengths_file = '/mnt/home/ssethuram/ceph/datasets/wave_tng.dat' # file containing wavelength array to load into np array    
######################################################################################

# #make parameter dist plots if you've run test already. Otherwise comment out.
# ms = np.load('stellarMassArray.npy')
# md = np.load('dustMassArray.npy')
# sfr = np.load('sfrArray.npy')
# mselist = np.loadtxt('Distributions/mselist_tng_dynamicmodel2.csv', delimiter=',', skiprows=3)
# mselist = mselist[:,1]
# try:
#     Path('./PARAMETERSPACEPLOTS').mkdir(parents=False, exist_ok=False)
# except Exception:
#     pass

# # print(ms.shape)
# # print(md.shape)
# # print(sfr.shape)
# # print(mselist.shape)
# cmap = plt.get_cmap('copper_r',150)
# norm=mpl.colors.Normalize(vmin=mselist.min(),vmax=0.08)
# sm = cm.ScalarMappable(norm=norm, cmap=cmap)


# # fig,ax=plt.subplots(1, figsize=(10,10))

# # tngplt = ax.scatter(np.log10(ms), np.log10(md), c=cmap(norm(mselist)), alpha= 0.9, cmap=cmap)
# # ax.set_xlabel('$\it{Log_{10}}$ (Stellar Mass / [ $M_{\odot}$ ])', fontsize=22, labelpad=15)
# # ax.set_ylabel('$\it{Log_{10}}$ (Dust Mass / [ $M_{\odot}$ ])', fontsize=22, labelpad=10)
# # ax.set_xticklabels(ax.get_xticks(), fontsize=18)
# # ax.set_yticklabels(ax.get_yticks(), fontsize=18)

# # ticklist = np.linspace(mselist.min(), 0.08, 15)
# # ticklist = [round(each, ndigits=2) for each in ticklist]
# # tick_font_size=16
# # cbar = fig.colorbar(sm, ticks=ticklist, ax = ax, format=mpl.ticker.ScalarFormatter(), pad=0, shrink = 1.0, fraction = 0.1, orientation = 'vertical')
# # cbar.set_label('MSE', labelpad = 10, size=20)
# # cbar.ax.tick_params(labelsize=tick_font_size)
# # # ax.set_title('M$_{dust}$ vs. M$_{star}$', pad=15, fontsize=25)
# # fig.savefig('./PARAMETERSPACEPLOTS/MDUSTvsMSTAR.png')

# # fig.clf()
# # plt.clf()

# # fig,ax=plt.subplots(1, figsize=(10,10))

# # tngplt = ax.scatter(np.log10(ms), np.log10(sfr), c=cmap(norm(mselist)), alpha= 0.8, cmap=cmap)
# # ax.set_xlabel('Stellar Mass / $\it{log_{10}}$ [ $M_{\odot}$ ]', fontsize=22, labelpad=15)
# # ax.set_ylabel('SFR / $\it{log_{10}}$ [ $M_{\odot}$ $yr^{-1}$]', fontsize=22, labelpad=10)
# # ax.set_xticklabels(ax.get_xticks(), fontsize=18)
# # ax.set_yticklabels(ax.get_yticks(), fontsize=18)

# # ticklist = np.linspace(mselist.min(), 0.08, 15)
# # ticklist = [round(each, ndigits=2) for each in ticklist]
# # tick_font_size=16
# # cbar = fig.colorbar(sm, ticks=ticklist, ax = ax, format=mpl.ticker.ScalarFormatter(), pad=0, shrink = 1.0, fraction = 0.1, orientation = 'vertical')
# # cbar.set_label('MSE', labelpad = 10, size=20)
# # cbar.ax.tick_params(labelsize=tick_font_size)
# # # ax.set_title('SFR vs. M$_{star}$, colored by MSE', pad=15, fontsize=25)
# # fig.savefig('./PARAMETERSPACEPLOTS/SFRvsMSTAR.png')

# # fig.clf()
# # plt.clf()

# # fig,ax=plt.subplots(1, figsize=(10,10))

# # tngplt = ax.scatter(np.log10(md), np.log10(sfr), c=cmap(norm(mselist)), alpha= 0.8, cmap=cmap)
# # ax.set_xlabel('Dust Mass / $\it{log_{10}}$ [ $M_{\odot}$ ]', fontsize=22, labelpad=15)
# # ax.set_ylabel('SFR / $\it{log_{10}}$ [ $M_{\odot}$ $yr^{-1}$]', fontsize=22, labelpad=10)
# # ax.set_xticklabels(ax.get_xticks(), fontsize=18)
# # ax.set_yticklabels(ax.get_yticks(), fontsize=18)

# # ticklist = np.linspace(mselist.min(), 0.08, 15)
# # ticklist = [round(each, ndigits=2) for each in ticklist]
# # tick_font_size=18
# # cbar = fig.colorbar(sm, ticks=ticklist, ax = ax, format=mpl.ticker.ScalarFormatter(), pad=0, shrink = 1.0, fraction = 0.1, orientation = 'vertical')
# # cbar.set_label('MSE', labelpad = 10, size=20)
# # cbar.ax.tick_params(labelsize=tick_font_size)
# # # ax.set_title('SFR vs. M$_{dust}$, colored by MSE', pad=15, fontsize=25)
# # fig.savefig('./PARAMETERSPACEPLOTS/SFRvsMDUST.png')

# # fig.clf()

# #COMBINED ONE
# fig,axs=plt.subplots(nrows = 1, ncols=3, figsize=(30,9.6))
# ticklist = np.linspace(mselist.min(), 0.08, 15)
# ticklist = [round(each, ndigits=2) for each in ticklist]
# tick_font_size=18

# axs[0].scatter(np.log10(ms), np.log10(md), c=cmap(norm(mselist)), alpha= 0.8, s=150, edgecolor='black', cmap=cmap)
# axs[0].set_xlabel('$\it{Log_{10}}$ (Stellar Mass / [ $M_{\odot}$ ])', fontsize=25, labelpad=15)
# axs[0].set_ylabel('$\it{Log_{10}}$ (Dust Mass /[ $M_{\odot}$ ])', fontsize=25, labelpad=10)
# axs[0].set_xticklabels(axs[0].get_xticks(), fontsize=18)
# axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=18)

# axs[1].scatter(np.log10(ms), np.log10(sfr), c=cmap(norm(mselist)), alpha= 0.8,s=150, edgecolor='black',  cmap=cmap)
# axs[1].set_xlabel('$\it{Log_{10}}$ (Stellar Mass / [ $M_{\odot}$ ])', fontsize=25, labelpad=15)
# axs[1].set_ylabel('$\it{Log_{10}}$ (SFR / [ $M_{\odot}$ $yr^{-1}$ ])', fontsize=25, labelpad=10)
# axs[1].set_xticklabels(axs[1].get_xticks(), fontsize=18)
# axs[1].set_yticklabels(axs[1].get_yticks(), fontsize=18)

# axs[2].scatter(np.log10(md), np.log10(sfr), c=cmap(norm(mselist)), alpha= 0.8,s=150, edgecolor='black',  cmap=cmap)
# axs[2].set_xlabel('$\it{Log_{10}}$ (Dust Mass / [ $M_{\odot}$ ])', fontsize=25, labelpad=15)
# axs[2].set_ylabel('$\it{Log_{10}}$ (SFR / [ $M_{\odot}$ $yr^{-1}$ ])', fontsize=25, labelpad=10)
# axs[2].set_xticklabels(axs[2].get_xticks(), fontsize=18)
# axs[2].set_yticklabels(axs[2].get_yticks(), fontsize=18)

# cbar = fig.colorbar(sm, ticks=ticklist, ax = axs[2], format=mpl.ticker.ScalarFormatter(), pad=0, shrink = 1.0, fraction = 0.1, orientation = 'vertical')
# cbar.set_label('MSE', labelpad = 10, size=25)
# cbar.ax.tick_params(labelsize=tick_font_size)
# fig.savefig('./PARAMETERSPACEPLOTS/COMBINED.pdf')

# fig.clf()

# STOP

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

fmodel = 'models/{}.pt'.format(fname) #make sure fmodel name matches the format for your model outputs
time1 = time.time()
# generate the architecture
model = architecture.dynamic_model2(input_size, output_size, numHL, hidden, dr)
model.to(device)    

# load best-model, if it exists
if os.path.exists(fmodel):  
    print('Loading model...')
    model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
else:
    raise Exception('Specified model does not exist: {}'.format(fmodel))

# define loss function
criterion = nn.MSELoss() 

# get the data
test_loader = data.create_dataset(mode, seed, fin, batch_size, shuffle=False)

# test the model
test_loss, points = 0.0, 0
model.eval()

# these are global properties I wanted to print on my test plots, so I'm tracking them. Change as necessary for your model
ogMS = []
ogMD = []
ogSFR = []
ogZ = []
ogYZ= []

# arrays containing test galaxy avg MSE and max fractional difference/region per test galaxy
listmse = []
fdiffs = []
maelist = []
mseBands = []
maeBands = []

# get wavelengths into logged and unlogged arrays
wavelength_unlogged = np.genfromtxt(wavelengths_file, dtype=None,delimiter=' ')
wavelengths_correct = np.log10(np.array(wavelength_unlogged))

nnestimatedSEDs = []
trueSEDs = []
fChiList = []


# create directory to store test plots
try:
    Path('./TestImsPDF').mkdir(parents=False, exist_ok=False)
except Exception as e:
    print(e)
try:
    Path('./TestImsPNG').mkdir(parents=False, exist_ok=False)
except Exception as e:
    print(e)

try:
    Path('./UnloggedTestIms').mkdir(parents=False, exist_ok=False)
except Exception as e:
    print(e)

# for test plot naming scheme
counter = -1
               
with torch.no_grad():
    # open test data; if larger than batch size, will iterate over x, y multiple times
    for x, y in test_loader:
        bs   = x.shape[0]  #batch size
        x, y = x.to(device), y.to(device)
        y_NN = model(x)
        test_loss += (criterion(y_NN, y).item())*bs
        points    += bs
        
        # iterate over each galaxy in test set
        for i in range(0, len(x)):
            counter += 1

            # sample of my denormalization function is in data.py. Modify for your dataset
#            denormed_sfr = data.denormSingle(fin, 'SFR', x[i][0], 'tng') 
#            denormed_ms = data.denormSingle(fin, 'M_star', x[i][2], 'tng')
#            denormed_md = data.denormSingle(fin, 'M_dust', x[i][1], 'tng')
#            denormed_Z = data.denormSingle(fin, 'metallicity', x[i][3], 'tng')
#            denormed_YZ = data.denormSingle(fin, 'y_metallicity', x[i][4], 'tng')

            # print(type(denormed_ms), type(denormed_md), type(denormed_md), type(denormed_Z), type(denormed_YZ))
            
#            ogMS.append(denormed_ms)
#            ogMD.append(denormed_md)
#            ogSFR.append(denormed_sfr)
#            ogZ.append(denormed_Z)
#            ogYZ.append(denormed_YZ)
            
            # get model input and output into np arrays
            try:
                nnestimate = model(x[i]).numpy()
                target = y[i].numpy()
            except:
                nnestimate = model(x[i]).cpu().numpy()
                target = y[i].cpu().numpy()
#            unloggedtarget = [10**each for each in target]
#            unloggedtng = [10**each for each in nnestimate]

#            nnestimatedSEDs.append(nnestimate)
#            trueSEDs.append(target)
                
            # get fractional diffs per wavelength to plot
#            fracdiff = data.calcFDiff(nnestimate, target, logged=True)
#            fChi = fracdiff['fracdiff']
#            fChiList.append(fChi)        

#            medae = median(np.abs(np.subtract(nnestimate,target)))
#            maelist.append(medae)

            #get mse loss for specific galaxy
#            summse = ((nnestimate - target)**2).mean() # can also use criterion here for shortform, this was a sanity check
#            listmse.append(float(summse))

            #get mse in diff bands
#            idx1 = np.where(wavelength_unlogged==0.400880633)[0][0]+1
#            idx2 = np.where(wavelength_unlogged==0.731680714)[0][0]
#            idx3 = np.where(wavelength_unlogged==3.072113)[0][0]+1
#            idx4 = np.where(wavelength_unlogged==51.709202399999995)[0][0]+1
#            idx5 = np.where(wavelength_unlogged==207.292178)[0][0]

#            m1 = ((nnestimate[:idx1] - target[:idx1])**2)
#            mse1 = (m1.mean())
#            mae1 = median(np.abs(np.subtract(nnestimate[:idx1], target[:idx1])))
#            m2 = ((nnestimate[idx1:idx2] - target[idx1:idx2])**2)
#            mse2 = m2.mean()
#            mae2 = median(np.abs(np.subtract(nnestimate[idx1:idx2], target[idx1:idx2])))
#            m3 = ((nnestimate[idx2:idx3] - target[idx2:idx3])**2)
#            mse3 = m3.mean()
#            mae3 = median(np.abs(np.subtract(nnestimate[idx2:idx3], target[idx2:idx3])))
#            m4 = ((nnestimate[idx3:idx4] - target[idx3:idx4])**2)
#            mse4 = m4.mean()
#            mae4 = median(np.abs(np.subtract(nnestimate[idx3:idx4], target[idx3:idx4])))
#            m5 = ((nnestimate[idx4:idx5] - target[idx4:idx5])**2)
#            mse5 = m5.mean()
#            mae5 = median(np.abs(np.subtract(nnestimate[idx4:idx5], target[idx4:idx5])))
#            m6 = ((nnestimate[idx5:] - target[idx5:])**2)
#            mse6 = m6.mean()
#            mae6 = median(np.abs(np.subtract(nnestimate[idx5:], target[idx5:])))
               
#            tempbands1 = [mse1, mse2, mse3, mse4, mse5, mse6]      
#            mseBands.append(tempbands1)
#            maeBands.append([mae1, mae2, mae3, mae4, mae5, mae6])

            # get max frac diff per wavelength region            
#            f1 = 0.
#            f2 = 0.
#            f3 = 0.
#            f4 = 0.
#            f5 = 0.
#            f6 = 0.
            ## change if/elif statements according to wavelength regions you want to specify
#            for idmse, wavelength in enumerate(wavelength_unlogged):
                # frac diff of that wavelength 
#                fnum = fChi[idmse]
#                if wavelength <= 0.4:
#                    #UV
#                    if np.abs(fnum) > np.abs(f1):
#                        f1 = fnum
#                    else:
#                        continue
#                elif wavelength <= 0.7:
#                    #Optical
#                    if np.abs(fnum) > np.abs(f2):
#                        f2 = fnum
#                    else:
#                        continue
#                elif wavelength <= 3.:
#                    #NIR
#                    if np.abs(fnum) > np.abs(f3):
#                        f3 = fnum
#                    else:
#                        continue
#                elif wavelength <= 50.:
#                    #PAH (3-12) & MIR
#                    if np.abs(fnum) > np.abs(f4):
#                        f4 = fnum
#                    else:
#                        continue
#                elif wavelength <=200.:
#                    #first part of FIR
#                    if np.abs(fnum) > np.abs(f5):
#                        f5 = fnum
#                    else:
#                        continue
#                else:
#                    #FIR pt 2
#                    if np.abs(fnum) > np.abs(f6):
#                        f6 = fnum
#                    else:
#                        continue
#            tempbands2 = [f1,f2,f3,f4,f5,f6]      
#            fdiffs.append(tempbands2)

            # avg frac diff of galaxy
#            meanfdiff = round(np.mean(fChi), 2)
        
             # set up test plot
            # fig, axs = plt.subplots(2, figsize=(12,13), sharex=True)#, constrained_layout=True)
            # # fig.suptitle('Predicted vs. True SED', fontsize=20, x=0.53)
            # # fig.tight_layout()

            # # plot true vs pred
            # axs[0].plot(wavelength_unlogged,target,'#09A7A5',linewidth=2, label='SKIRT output (True)')
            # axs[0].plot(wavelength_unlogged,nnestimate,'#2F3534',linewidth=1,label='NN estimate (Predicted)')
            # props = dict(boxstyle='round', pad=1, facecolor = '#09A7A5', edgecolor='black', alpha=0.3)
            # axs[0].text(wavelength_unlogged.min()+0.01, target.min()-0.8, sci_notation('SFR', '[$M_{\odot}$ $yr^{-1}$]', str(denormed_sfr), sig_fig=2) + '\n' + sci_notation('$M_{dust}$', '[$M_{\odot}$]', str(denormed_md), sig_fig=2) + '\n' + sci_notation('$M_{star}$', '[$M_{\odot}$]', str(denormed_ms), sig_fig=2), fontsize=20,bbox=props)
            # axs[0].set_ylim([nnestimate.min()-1,nnestimate.max()+1])
            # # axs[0].set_xlabel(r'Log$_{10}$ ($\lambda_{\rm{rest}}$) / [ $\mu m$ ]',fontsize=18, labelpad=12)
            # axs[0].set_ylabel(r'Log$_{10}$ ($\lambda*F_{\lambda}$) / [ $\itW$ $m^{-2}$ ]',fontsize=25, labelpad=7)
            # # axs[0].set_xticks([-1,-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3], fontsize = 18)
            
            # ylabels = np.arange(int(min(nnestimate.min(), target.min())-1), int(max(nnestimate.max(), target.max())+1), 1)
            # axs[0].set_yticks(ylabels)
            # print(ylabels)
            # axs[0].set_yticklabels(ylabels,fontsize=22)
            # axs[0].set_title('MSE : {:.4f}; MAE: {:.4f}'.format(summse, medae), pad=5, fontsize=30)
            
            # axs[1].plot(wavelength_unlogged,fChi,'#2F3534', label='Fractional Difference')
            # axs[1].plot([0.1,1000], [meanfdiff, meanfdiff], '#2F3534', linestyle='dashed', linewidth=1, label = 'Average Fractional Difference')
            # axs[1].plot([0.1,1000],[0,0],color='green',linestyle='dashed',linewidth=4, label='Predicted = True')
            
            # axs[1].set_xlabel(r'$\lambda_{\rm{rest}}$ / Log$_{10}$ [ $\mu m$ ]',size=30, labelpad=7)
            # axs[1].set_ylabel(r'Log$_{10}$ [ ${Predicted}$ / ${True}$ ]',size=25)
            # # axs[1].set_xticks([0.1, 0.3, 1, 3, 10, 31, 100, 2.5, 3], fontsize = 18)
            # axs[1].set_yticks(axs[1].get_yticks())
            # axs[1].set_yticklabels(axs[1].get_yticklabels(),fontsize=22)
            # axs[1].set_xticks(axs[1].get_xticks())
            # axs[1].set_xticklabels(axs[1].get_xticklabels(),fontsize=22)
            # axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # axs[1].set_xscale('log')
            # axs[0].set_xscale('log')
            # # axs[1].set_title('Fractional difference (dex)', pad=5, fontsize=15)
            # # axs[1].text(wavelengths_correct.max()-1, (min(fChi)+np.abs(0.2*min(fChi))), 'MSE: {:.4f}'.format(summse), fontsize=11, bbox=props1)
            
            # axs[0].legend(fontsize=20)
            # axs[1].legend(fontsize=20)
            
            # plt.gcf()
            # plt.subplots_adjust(left=0.12, right = 0.95, top=0.95, bottom = 0.1, hspace=0.1)
            # # plt.subplots_adjust(bottom=5)
            
            # fig.savefig("./TestImsPDF/TestNN_{}.pdf".format(counter))
            # fig.savefig("./TestImsPNG/TestNN_{}.png".format(counter))
            # # fig.clf()

        #     # plot unlogged seds
        #     # plt.clf()
        #     # plt.plot(wavelength_unlogged,unloggedtng,'r-',label='TNG NN estimate')           
        #     # plt.plot(wavelength_unlogged,unloggedtarget,'b-',label='SKIRT output')
        #     # props = dict(boxstyle='round', facecolor = 'mistyrose', alpha=0.4)
        #     # plt.text(10, min(unloggedtarget)+0.1*min(unloggedtarget), 'SFR: %s $M_{\odot} yr^{-1}$\n$M_{dust}$: %s $M_{\odot}$\n$M_{star}$: %s $M_{\odot}$' % (denormed_sfr, denormed_md, denormed_ms), fontsize=11,bbox=props)
        #     # plt.text(1, max(unloggedtarget), 'MSE: {:.4f}'.format(summse), fontsize=11, bbox=props)
        #     # plt.xlabel(r'$\lambda_{\rm{rest}}/ [ \mu m$ ]',size=16)
        #     # plt.ylabel(r'$\lambda*F_{\lambda}/ [ Wm^{-2}$ ]',size=16)
        #     # plt.xscale('log')
        #     # plt.legend()
        #     # plt.savefig('./UnloggedTestIms/TestNN_{}.png'.format(counter))
        

        print ('shape of each output')
        print (np.shape(x))
        print (np.shape(y))
        print (np.shape(y_NN))

test_loss /= points
print('Test loss:', test_loss)
time2 = time.time()
print(time2-time1)
stop
# nnestimatedSEDs = np.array(nnestimatedSEDs)
# trueSEDs = np.array(trueSEDs)
np.save('nnestimatedSEDs.npy', nnestimatedSEDs)
np.save('trueSEDs.npy', trueSEDs)
np.save('fChilist.npy', fChiList)

### below is a bunch of extra-inefficient code meant to help for analysis; 
### delete/comment if you'd like to write your own analysis

ogMS = np.array(ogMS).astype(float)
ogSFR = np.array(ogSFR).astype(float)
ogMD = np.array(ogMD).astype(float)
ogZ = np.array(ogZ).astype(float)
ogYZ = np.array(ogYZ).astype(float)

np.save('stellarMassArray', ogMS)
np.save('dustMassArray', ogMD)
np.save('sfrArray', ogSFR)
np.save('ZArray', ogZ)
np.save('YZArray', ogYZ)

# make directory containing files/plots for further analysis
try:
    Path('./Distributions').mkdir(parents=False, exist_ok=False)
except Exception as e:
    print(e)

# save mses to csv
avgmse = np.mean(listmse)
f = open('./Distributions/mselist_{}.csv'.format(fname), 'w')
f.write('Average MSE over test set:, {}\n'.format(avgmse))
f.write('Test loss: {}\n'.format(test_loss))
f.write('index, MSE\n')
for id, each in enumerate(listmse):
    f.write("{}, {}".format(str(id), str(each)))
    f.write('\n')
f.close()    

#save mse per band to csv
f = open('./Distributions/mseBandList_{}.csv'.format(fname), 'w')
f.write('halo id, UV, Optical, NIR, MIR, FIR 1, FIR 2\n')
for id, each in enumerate(mseBands):
    f.write('{}, {}'.format(str(id), (str(each).strip('[]'))))
    f.write('\n')
f.close()

#save mae per band to csv
f = open('./Distributions/maeBandList_{}.csv'.format(fname), 'w')
f.write('halo id, UV, Optical, NIR, MIR, FIR 1, FIR 2\n')
for id, each in enumerate(maeBands):
    f.write('{}, {}'.format(str(id), (str(each).strip('[]'))))
    f.write('\n')
f.close()

# save frac diffs to csv
f = open('./Distributions/maxfdifflist_{}.csv'.format(fname), 'w')
f.write('halo id, Max FDiff UV, Max FDiff Optical, Max FDiff NIR, Max FDiff MIR, Max FDiff FIR 1, Max FDiff FIR 2\n')
for id, each in enumerate(fdiffs):
    f.write('{}, {}'.format(str(id), (str(each).strip('[]'))))
    f.write('\n')
f.close()

f = open('./Distributions/medabserror_{}.csv'.format(fname), 'w')
for id, each in enumerate(maelist):
    f.write('{}, {}'.format(str(id), (str(each))))
    f.write('\n')
f.close()

first = 0.
second = 0.
third = 0.
fourth = 0.
fifth = 0.
sixth = 0.
seventh = 0.
eighth = 0.
last = 0.
totnum = (len(listmse))

#each row is a different threshold, each column is the next band. the number represent how many halos have max fracdiff/mse in that bucket
# fdiffbins = [[0,0,0,0,0],
#                 [0,0,0,0,0],
#                 [0,0,0,0,0],
#                 [0,0,0,0,0],
#                 [0,0,0,0,0],
#                 [0,0,0,0,0],
#                 [0,0,0,0,0]]
# msebins = [[0,0,0,0,0],
#            [0,0,0,0,0],
#            [0,0,0,0,0],
#            [0,0,0,0,0],
#            [0,0,0,0,0],
#            [0,0,0,0,0],
#            [0,0,0,0,0]]

# fdiffthresholds = [0.10, 0.25, 0.4, 0.6, 0.75, 1,100]
# msethresholds = [0.01,0.02,0.03,0.04,0.05,0.06,100]

# c = 0
# for hid, halo in enumerate(fdiffs):
#     for colid, fdiffband in enumerate(halo):
#         if fdiffband <= fdiffthresholds[0]:
#             fdiffbins[0][colid] += 1
#             c+=1
            
#         elif fdiffband <= fdiffthresholds[1]:
#             fdiffbins[1][colid] += 1
#             c+=1

#         elif fdiffband <= fdiffthresholds[2]:
#             fdiffbins[2][colid] += 1
#             c+=1

#         elif fdiffband <= fdiffthresholds[3]:
#             fdiffbins[3][colid] += 1
#             c+=1

#         elif fdiffband <= fdiffthresholds[4]:
#             fdiffbins[4][colid] += 1
#             c+=1

#         elif fdiffband <= fdiffthresholds[5]:
#             fdiffbins[5][colid] += 1
#             c+=1

#         else:
#             fdiffbins[6][colid] += 1
#             c+=1
# print(fdiffbins)
# print(c)

# for mid, num in enumerate(listmse):
#     #now let's do the percentages
#     if num <= 0.005:
#         first += 1
#     elif num <= 0.01:
#         second += 1
#     elif num <= 0.02:
#         third += 1
#     elif num <= 0.03:
#         fourth += 1
#     elif num <= 0.04:
#         fifth += 1
#     elif num <= 0.05:
#         sixth += 1
#     elif num <= 0.06:
#         seventh += 1
#     elif num <= 0.1:
#         eighth += 1
#     else:
#         last += 1
        
# first = round(first/totnum, 2)
# second = round(second/totnum, 2)
# third = round(third/totnum, 2)
# fourth = round(fourth/totnum, 2)
# fifth = round(fifth/totnum, 2) 
# sixth = round(sixth/totnum, 2)
# seventh = round(seventh/totnum, 2)
# eighth = round(eighth/totnum, 2)
# last = round(last/totnum, 2) 
# print(first+second+third+fourth+fifth+sixth+seventh+eighth+last)
                    
# plt.plot(listmse, 'o')
# plt.title('MSE distribution for test {}'.format(fname))
# plt.yscale('log')
# plt.ylabel('MSE values', fontsize=18)
# plt.xlabel('halo #', fontsize=18)
# fig = plt.gcf()
# fig.set_size_inches(11,8.5)
# fig.savefig('./Distributions/mseDistribution_{}.png'.format(fname))
# plt.clf()

# msesarr = [first,second,third,fourth,fifth,sixth,seventh]
# mses = {'mse < 0.005': first, '0.005 < mse <= 0.01': second, '0.01 < mse <= 0.02': third, '0.02 < mse <= 0.03': fourth, '0.03 < mse <= 0.04': fifth, '0.04 < mse <= 0.05': sixth, '0.05 < mse <= 0.06': seventh, '0.06 < mse <= 0.1': eighth, '0.1 < mse': last}  

# arr = plt.hist(listmse, edgecolor='black', linewidth=1.2, bins=[0,0.005,0.01,0.02,0.03,0.04,0.05,0.06])
# maxy = arr[0].max() + 5
# for i in range(7):
#     xax = arr[1][i] 
#     yax = arr[0][i]+2
#     plt.text(xax, yax, '{}%'.format(round(msesarr[i],1)), va='bottom', ha='center',fontsize=14)

# plt.text(((arr[1].max()-arr[1].min())/2), arr[0].max(), 'Max MSE 0.1-1.0 micron: {}\nMax MSE 1.0-10 micron: {}\nMax MSE 10-100 micron: {}\nMax MSE 100-300 micron: {}\nMax MSE 300-1000 micron: {}'.format(max1,max2,max3,max4,max5))
# plt.title('MSE distribution stats')
# plt.xlim([0,0.1])
# plt.ylabel('Num halos in MSE bin', fontsize=16)
# plt.xlabel('MSE value bin', fontsize=16)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# fig = plt.gcf()
# fig.set_size_inches(14,10)
# fig.savefig('./Distributions/mseDistHist_{}.png'.format(fname))
# plt.clf()

#Now plot more fun stuff but in group herd format

minMS = min(ogMS)
maxMS = max(ogMS)
minSFR = min(ogSFR)
maxSFR = max(ogSFR)
minMD = min(ogMD)
maxMD = max(ogMD)
minZ = min(ogZ)
maxZ = max(ogZ)
minYZ = min(ogYZ)
maxYZ = max(ogYZ)

plt.clf()
plt.scatter((ogMS), (ogZ), c=((listmse)), vmin=(min(listmse)),vmax=((np.mean(listmse))))
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'Z$_{star}$ / $log$ ($M_{Z}$ / $M_{\odot}$)', fontsize=18)
plt.xlabel(r'$M_{star} / $log$ ($M_{\odot}$)', fontsize=18)
cbar = plt.colorbar()
cbar.set_label('MSE', size=18)
cbar.ax.tick_params(labelsize=16)
fig = plt.gcf()
fig.set_size_inches(12,8)
plt.savefig('ZvsMS.pdf')

plt.clf()
plt.scatter((ogMD), (ogZ), c=((listmse)), vmin=(min(listmse)),vmax=((np.mean(listmse))))
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'Z$_{star}$ / $log$ ($M_{Z}$ / $M_{\odot}$)', fontsize=18)
plt.xlabel(r'$M_{dust} / $log$ ($M_{\odot}$)', fontsize=18)
cbar = plt.colorbar()
cbar.set_label('MSE', size=18)
cbar.ax.tick_params(labelsize=16)
fig = plt.gcf()
fig.set_size_inches(12,8)
plt.savefig('ZvsMD.pdf')

plt.clf()
plt.scatter((ogMS), (ogYZ), c=((listmse)), vmin=(min(listmse)),vmax=((np.mean(listmse))))
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'Z$_{star < 10 Myr}$ / $log$ ($M_{Z}$ / $M_{\odot}$)', fontsize=18)
plt.xlabel(r'M$_{star}$ / $log$ ($M_{\odot}$)', fontsize=18)
cbar = plt.colorbar()
cbar.set_label('MSE', size=18)
cbar.ax.tick_params(labelsize=16)
fig = plt.gcf()
fig.set_size_inches(12,8)
plt.savefig('YZvsMS.pdf')

plt.clf()
plt.scatter((ogMD), (ogYZ), c=((listmse)), vmin=(min(listmse)),vmax=((np.mean(listmse))))
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'Z$_{star < 10 Myr}$ / $log$ ($M_{Z}$ / $M_{\odot}$)', fontsize=18)
plt.xlabel(r'M$_{dust}$ / $log$ ($M_{\odot}$)', fontsize=18)
cbar = plt.colorbar()
cbar.set_label('MSE', size=18)
cbar.ax.tick_params(labelsize=16)
fig = plt.gcf()
fig.set_size_inches(12,8)
plt.savefig('YZvsMD.pdf')

plt.clf()
plt.scatter((ogMS), (ogSFR), c=((listmse)), vmin=(min(listmse)),vmax=((np.mean(listmse))))
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'SFR / log ($M_{\odot} yr^{-1}$)', fontsize=18)
plt.xlabel(r'$M_{star} / log (M_{\odot}$)', fontsize=18)
cbar = plt.colorbar()
cbar.set_label('MSE', size=18)
cbar.ax.tick_params(labelsize=16)
fig = plt.gcf()
fig.set_size_inches(12,8)
plt.savefig('SFRvsMS.pdf')


plt.clf()
plt.scatter((ogMD), (ogSFR), c=(listmse), vmin=(float(min(listmse))),vmax=((np.mean(listmse))))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_{dust}/M_{\odot}$', fontsize=18)
plt.ylabel(r'SFR/$M_{\odot} yr^{-1}$', fontsize=18)
cbar = plt.colorbar()
cbar.set_label('MSE', size=18)
cbar.ax.tick_params(labelsize=16)
fig = plt.gcf()
fig.set_size_inches(12,8)
fig.savefig('SFRvsMD.pdf')

plt.clf()
plt.scatter((ogMD), (ogMS), c=(listmse), vmin=(float(min(listmse))),vmax=((np.mean(listmse))))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_{dust}/M_{\odot}$', fontsize=18)
plt.ylabel(r'$M_{star}/M_{\odot}$', fontsize=18)
cbar = plt.colorbar()
cbar.set_label('MSE', size=18)
cbar.ax.tick_params(labelsize=16)
fig = plt.gcf()
fig.set_size_inches(12,8)
fig.savefig('MDvsMS.pdf')

#plot losses and print min loss values on them
losses = np.loadtxt('losses/{}.txt'.format(fname), skiprows=12)

epoch = losses[:,0]
trainloss = losses[:,1]
validloss = losses[:,2]
testloss = losses[:,3]
lossArray = [trainloss, validloss, testloss]

maxfinalloss = -1
        
for a in lossArray:
    finalminval = a.min()
    if finalminval > maxfinalloss:
        maxfinalloss = finalminval
    else:
        continue
        
plt.clf()
plt.plot(epoch, trainloss, label = 'train loss')
plt.plot(epoch, validloss, label = 'valid loss')
plt.plot(epoch, testloss, label = 'test loss')
plt.ylim([0, maxfinalloss+10])
props = dict(boxstyle='round', facecolor = 'blue', alpha=0.3)
plt.text(epoch.max()-1200, maxfinalloss+5, 'min train loss: {}\nmin valid loss: {}\nmin test loss: {}'.format(trainloss.min(), validloss.min(), testloss.min()), fontsize=11,bbox=props)

plt.legend()

fig = plt.gcf()
fig.set_size_inches(10,5)
fig.savefig('lossesPlotted_{}.png'.format(fname))
