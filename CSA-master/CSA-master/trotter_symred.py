import numpy as np
import os
import scipy as sp
import itertools
import openfermion
path_prefix="../CSA-master/" ##Top of the project, corresponding to CSA/
import sys
sys.path.append(path_prefix)
import time 
from scipy.fft import fft, ifft, ifftshift, fftfreq
from scipy.signal import chirp, find_peaks, peak_widths
import matplotlib.pyplot as plt
from scipy import linalg
import pickle
import saveload_utils as sl
def trotter(method,mol,gs,deltaT):
    start = time.time()
    print("Start")
    #Creating Directory for the molecule to store results, uncomment when needed
    if not os.path.isdir('./Results/'+mol):
        os.mkdir('./Results/'+mol+'/')
    print("Running for method:",method)
    #symmetry-adapted wavefunction
    hf = np.load("./SymFrags/"+mol+"_SymHf.npy")
   
    ## Defining control parameters
    # Total Time of propagation
    T = 10000000
    m = np.arange(1,11,1)
    # trotter steps
    deltat = deltaT/m
    # discrete evolution times
    t = np.arange(0,T,deltaT)

    # Changes in Ground state energy due to trotterization for different trotter steps
    trotter_error = [] # stores the error in the ground state energy value due to trotterization for different trotter steps
    for k in range(len(deltat)):
        print("Start")
        print("Running for data point",k)
        # trotter number 
        n = np.arange(0,T/deltat[k],deltaT/deltat[k])
        n = n.astype(int)
        # constructing propagator dt
        x = np.load('./MatrixFrags/'+mol+'/'+method+'/0.npy')
        trotter_state = linalg.expm(-1j*x*deltat[k])
        for j in np.arange(1,3,1): # modify the length as per the number  of fragments of the respective method
            #print(j)
            x = np.load("./MatrixFrags/"+mol+"/"+method+"/"+str(j)+".npy")
            trotter_state = linalg.expm(-1j*x*deltat[k])@trotter_state
        w, v = sp.linalg.eig(trotter_state)
        u = np.stack(v)
        print(u.shape)
        u_dag = (u.conj()).T
        print(u_dag.shape)
        print(hf.shape)
        phi = u_dag@hf.T
        c= np.ravel((np.multiply(phi.conj(),phi)).T)
        print(c.shape)
        new_c = [] # storing the significant elements of c_sq vector
        d = [] # storing the corresponding elements of the D vector
        for i in range((c).shape[0]):
            if  c[i] >= 1e-6:  # filtering out the significant terms
                new_c.append(c[i])
                d.append(w[i])
        print(len(d))
        print(len(new_c))
        #product_state = []  # propagator(t_i)
        trotter_expectation = [] # autocorrelation fucntion
        product_state = (d**n[0])
        trotter_expectation.append(np.inner(new_c,product_state))
        for i in range(len(n)-1):
            product_state = (np.multiply(product_state,(d)**(n[i+1]-n[i])))
            trotter_expectation.append(np.inner(new_c,product_state))
        # fourier transform
        xf_trotter = 2*np.pi*fftfreq(len(t),deltaT)
        yf_trotter = ifft(np.array(trotter_expectation))   
        peaks_trotter, _ = find_peaks(np.abs(yf_trotter))
        results_half_trotter = peak_widths(np.abs(yf_trotter), peaks_trotter, rel_height=0.5)
        trotter_error.append( min(xf_trotter[peaks_trotter])-gs)
        print(trotter_error[k])
    
    results = {}
    total_time = time.time()-start
    total_time = total_time/60 # in hrs
    dt_sq = deltat**2
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(dt_sq, trotter_error)
    results['slope'] = slope
    results['dt_sq'] = dt_sq
    results['r_value']= r_value
    results['trotter_error']= trotter_error
    results['total_time'] = total_time
    f=open('./Results/'+mol+'/'+method,'wb')
    pickle.dump(results,f)
    f.close()
# will generate an error if directory already exists
#if not os.path.isdir('./Results/'):
#    os.mkdir('./Results/')
# importing femrion hamiltonian
mol = "h2" # options: h2, lih, beh2, h2o, nh3
method = "FRO" # options: svd, SVDLCU, FRO, GFRO, GFROLCU, GCSASD
gs = np.load('./SymFrags/'+mol+'_gs.npy')
# Sample Spacing i.e, T/deltaT is the No. of points fed to the fourier tranform
deltaT = 2 # change with molecule # 2 for h2, 0.39 for lih, 0.2 for beh2, 0.04 for h2o, 0.05 for nh3 
# generating results for fermionic based partitioning methods 
trotter(method,mol,gs,deltaT)

