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

def trotter(method, n_qubits, gs, constant):
    start = time.time()
    print("Start")
    #if not os.path.isdir('./Results/NH3'):
    #    os.mkdir('./Results/NH3/')
    # Importing Fragments
    f = open("./Tapered/nh3_"+method+"_jw_TapFrags", 'rb')
    dict = pickle.load(f)
    f.close()
    ListFrags=dict['grouping'] # stores all the fragments
    number_frags = len(ListFrags) # Number of fragments
    ListFrags[0] += openfermion.QubitOperator("",constant ) # adding the constant term
    
    ## Defining control parameters
    # Total Time of propagation
    m = np.arange(1,11,1)
    deltaT = 0.05
    # trotter steps
    deltat = deltaT/m
    # discrete evolution times

    # Changes in Ground state energy due to trotterization for different trotter steps
    trotter_error = [] # stores the error in the ground state energy value due to trotterization for different trotter steps
    for k in range(len(deltat)):
        print("Start")
        print("Running for data point",k)
        # trotter number 
        # constructing propagator dt
        jw_ham_sp = openfermion.linalg.get_sparse_operator(ListFrags[0],n_qubits)
        trotter_state = linalg.expm(-1j*jw_ham_sp*deltat[k])
        for j in range(len(ListFrags)-1):
            print(j)
            jw_ham_sp = openfermion.linalg.get_sparse_operator(ListFrags[j+1],n_qubits)
            trotter_state = linalg.expm(-1j*jw_ham_sp*deltat[k])@trotter_state
        # diagonalizaion of the propagator with eigendecomposition
        w,v = sp.linalg.eig(trotter_state.todense())
        z = (np.log(w)/(-1j*deltat[k])).real
        print(min(z))
        trotter_error.append(min(z)-gs)
        print(trotter_error[k])
    results = {}
    total_time = time.time()-start
    total_time = total_time/3600 
    # in hrs
    dt_sq = deltat**2
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(dt_sq, trotter_error)
    results['slope'] = slope
    results['dt_sq'] = dt_sq
    results['r_value']= r_value
    results['trotter_error']= trotter_error
    results['number_frags'] = number_frags
    results['total_time'] = total_time
    f=open('./Results/NH3/'+method,'wb')
    pickle.dump(results,f)
    f.close()
    print("Done executing after",total_time, "hrs")

h_ferm=sl.load_fermionic_hamiltonian("nh3",path_prefix) # fermionic hamiltonian
n_qubits = openfermion.count_qubits(h_ferm) # no. of qubits
h_jw = openfermion.transforms.jordan_wigner(h_ferm) # jordan wigner transformatio for qubit operator
# eigenspectrum of the hamiltonian
spectrum = openfermion.eigenspectrum(h_jw)
gs = min(spectrum) # ground state energy
constant = h_jw.constant 
# generating results for qubit based partitioning methods (options: fc, qwc, fc_si, qwc_si)
# coment / uncomment these lines for the required methods
trotter("fc",n_qubits,gs,constant)
#trotter("fc_si",n_qubits,gs,constant)
#trotter("qwc", n_qubits,gs,constant)
#trotter("qwc_si",n_qubits,gs,constant)