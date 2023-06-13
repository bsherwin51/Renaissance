
# coding: utf-8

# In[1]:


#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import ytree
import yt
import cPickle as pickle
import os
import math


# In[2]:


from matplotlib import rc, rcParams
rc('font', size=18)
rc('xtick', direction='in')
rc('ytick', direction='in')


# # Read halo merger trees

# In[3]:


trees = ytree.load('rockstar_halos/trees/tree_0_0_0.dat')
trees.add_alias_field('mass', 'Mvir')


# In[20]:


mrange = [1e8, 1e10]
halo_mmp = []
for tree in trees:
    if tree['mass'].in_units('Msun') < mrange[0] or tree['mass'].in_units('Msun') > mrange[1]:
        continue
    mmp = dict(center=[], mass=[], redshift=[], rvir=[])
    for prog in tree.pwalk():
        pos = yt.YTArray([prog['x'], prog['y'], prog['z']]) / trees.box_size
        mmp['center'].append(pos)
        mmp['mass'].append(prog['mass'].in_units('Msun'))
        mmp['redshift'].append(prog['redshift'])
        mmp['rvir'].append(prog['virial_radius'] / trees.box_size)
    mmp['center'] = yt.YTArray(mmp['center'])
    mmp['mass'] = yt.YTArray(mmp['mass'])
    mmp['redshift'] = yt.YTArray(mmp['redshift'])
    mmp['rvir'] = yt.YTArray(mmp['rvir'])
    halo_mmp.append(mmp)


# In[21]:


print halo_mmp[0]['rvir']


# # Read simulation redshifts

# In[6]:


pfn = []
redshifts = []
if os.path.exists('redshifts.dat'):
    lines = open('redshifts.dat').readlines()
    for l in lines:
        pfn.append(l.split(':')[0])
        redshifts.append(float(l.split('=')[-1]))
else:
    ts = yt.DatasetSeries('DD????/output_????')
    for ds in ts:
        redshifts.append(ds.current_redshift)
        pfn.append(ds.parameter_filename)
    
redshifts = np.array(redshifts)
pfn = np.array(pfn)
isort = redshifts.argsort().astype('int')[::-1]
redshifts = redshifts[isort]
pfn = pfn[isort]


# # Read halo properties

# In[22]:


print halo_mmp[0]['rvir']


# In[23]:


pfile = 'fesc.cpkl'
if os.path.exists(pfile):
    fp = open(pfile, 'rb')
    catalog = pickle.load(fp)
    fp.close()
else:
    catalog = []
    for fn in pfn:
        entry = {}
        fp = h5.File('fesc-data/p%s_fesc_halos.h5' % (fn.split('/')[1]), 'r')
        nhalos = len(fp.keys())
        if nhalos == 0:
            catalog.append(entry)
            fp.close()
            continue
        gp0 = fp.values()[0]
        avoid = ['Pop2_Stars', 'Pop3_Stars', 'fesc_bins', 'fesc_hist']
        vector_fields = ['center']
        # Create empty entry
        for f in gp0:
            if f not in avoid:
                if f in vector_fields:
                    entry[f] = np.zeros((nhalos,3))
                else:
                    entry[f] = np.zeros(nhalos)
        # Read the data
        i = 0
        for g in fp.itervalues():
            for f in g:
                if f not in avoid:
                    entry[f][i] = g[f].value
            i += 1
        catalog.append(entry)
        fp.close()
    # Serialize properties
    fp = open(pfile, 'wb')
    pickle.dump(catalog, fp)
    fp.close()


# # Link halo properties to merger tree nodes

# In[24]:


def find_nearest(array, value):
    return np.abs(array - value).argmin()


# In[25]:


print halo_mmp[0]['rvir']


# In[26]:


fields = catalog[-1].keys()
fields.remove('center')
fields.remove('rvir')
for halo in halo_mmp:
    nnode = len(halo['redshift'])
    for f in fields:
        halo[f] = np.zeros(nnode)
    for inode in range(nnode):
        ip = find_nearest(redshifts, halo['redshift'][inode])
        
        # dr = separation, R = radius of target halo, r = radii of all catalog halos
        dr = np.sqrt(((halo['center'][inode] - catalog[ip]['center'])**2).sum(1))
        R = halo['rvir'][inode]
        r = catalog[ip]['rvir'] * (1 + redshifts[ip]) / trees.box_size.in_units('kpc').v
        
        # Calculate the sphere-sphere intersection. The maximum is the match.
        # http://mathworld.wolfram.com/Sphere-SphereIntersection.html
        vol = np.pi * (R+r-dr)**2 * (dr**2 + 2*dr*r - 3*r**2 + 2*dr*R + 6*r*R - 3*R**2) / (12*dr)
        vol[dr > r+R] = 0.0  # No intersection
        vol[dr < R-r] = (4*np.pi/3) * r[dr < R-r]**3  # Complete intersection
        
        match = vol.argmax()
        rvir = catalog[ip]['rvir'][match] * (1 + redshifts[ip]) / trees.box_size.in_units('kpc').v
        # Check whether separation is within the virial radius
        if dr[match] >= rvir:
            continue
        # Add halo properties to mmp history
        for f in fields:
            halo[f][inode] = catalog[ip][f][match]
    # Calculate SFR (last 20 Myr) [Msun/yr]
    halo['SFR'] = halo['Mstar_pop2_young'] / 20e6
    halo['fesc'] = np.maximum(halo['fesc'], 1e-6)


# In[27]:


print halo['mass']


# # Save to file for plotting

# In[28]:


fp = h5.File('catalog.h5', 'w')
for i,halo in enumerate(halo_mmp):
    hname = "Halo%8.8d" % (i)
    gp = fp.create_group(hname)
    for k,v in halo.items():
        gp[k] = v
fp.close()


# In[29]:


1 / (halo_mmp[0]['redshift']+1)


# In[30]:


halo_mmp[0]['mass']


# In[31]:


len(halo_mmp)


# In[ ]:


fields = ['mass', 'Mstar', 'SFR', 'fesc', 'fgas', 'fstar']
log = [True, True, True, True, False, True]
ylabel = [r'M$_{\rm vir}$ [M$_\odot$]', r'M$_{\star}$ [M$_\odot$]', 'SFR [M$_\odot$/yr]',
          r'f$_{\rm esc}$', r'f$_{\rm gas}$', r'f$_\star$']
nhalos = len(halo_mmp)
for i in range(nhalos):
    if halo_mmp[i]['Mstar'][0] < 1: continue
    print "Plotting Halo %06d of %06d" % (i+1, nhalos)
    fig, ax = plt.subplots(6, 1, figsize=(8,16), sharex=True)
    plt.subplots_adjust(hspace=1e-3)
    for j in range(len(fields)):
        if log[j]:
            ax[j].semilogy(halo_mmp[i]['redshift'], halo_mmp[i][fields[j]], lw=3)
        else:
            ax[j].plot(halo_mmp[i]['redshift'], halo_mmp[i][fields[j]], lw=3)
        ax[j].set_ylabel(ylabel[j])
    ax[0].set_title('Halo %d' % (i))
    ax[-1].set_xlabel('Redshift')
    plt.savefig('halo%6.6d-evo.png' % (i))
    del fig
    del ax
