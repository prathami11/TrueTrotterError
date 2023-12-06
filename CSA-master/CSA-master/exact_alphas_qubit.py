# code to caluclate "exact" alphas
import numpy as np
import os
import scipy as sp
import itertools
import openfermion
path_prefix="../CSA-master/" ##Top of the project, corresponding to CSA/
import sys
sys.path.append(path_prefix)
import time 
from scipy import linalg
import pickle
import saveload_utils as sl

def trotter(method,mol, h_jw, n_qubits,constant,t):
    start = time.time()
    print("Start")
    # Creating Directory for the molecule to store results, uncomment when needed
    #if not os.path.isdir('./Exact_Alphas/'+mol):
    #    os.mkdir('./Exact_Alphas/'+mol+'/')
    # Importing Fragments
    f = open("./Frag_Lib/"+method+"/"+mol+"_"+method+"Frags", 'rb')
    dict = pickle.load(f)
    f.close()
    ListFrags=dict['grouping'] # stores all the fragments
    number_frags = len(ListFrags) # Number of fragments
    ListFrags[0] += openfermion.QubitOperator("",constant ) # adding the constant term
    #Exact propagator
    H = openfermion.linalg.get_sparse_operator(h_jw,n_qubits)
    exact_state= linalg.expm(-1j*H*t)
    print("Exact State Done")
    # Trotterized propagator
    jw_ham_sp = openfermion.linalg.get_sparse_operator(ListFrags[0],n_qubits)
    trotter_state = linalg.expm(-1j*jw_ham_sp*t)
    for j in range(len(ListFrags)-1):
        print(j)
        jw_ham_sp = openfermion.linalg.get_sparse_operator(ListFrags[j+1],n_qubits)
        trotter_state = linalg.expm(-1j*jw_ham_sp*t)@trotter_state
    print("trotterized state done")
    w,v = sp.linalg.eig(exact_state.todense()-trotter_state.todense())
    z = max(abs(w))
    error = z/t**2
    print(error)
    results = {}
    total_time = time.time()-start
    total_time = total_time/3600 
    # in hrs
    dt_sq = t**2
    results['alpha'] = error
    results['dt_sq'] = t**2
    results['number_frags'] = number_frags
    results['total_time'] = total_time
    f= open('./Exact_Alphas/'+mol+'/'+method,'wb')
    pickle.dump(results,f)
    f.close()
    print("Done executing after",total_time, "hrs")
# will generate an error if directory already exists
#if not os.path.isdir('./Exact_Alphas/'):
#    os.mkdir('./Exact_Alphas/')
t=1e-5 # time step
method = "fc" # options: qwc, qwc_si, fc, fc_si
mol = "h2" # option: h2, lih, beh2, h2o, nh3
# hamiltonian
h_ferm=sl.load_fermionic_hamiltonian(mol,path_prefix) # fermionic hamiltonian wrt molecule
h_jw = openfermion.transforms.jordan_wigner(h_ferm)
constant = h_jw.constant
n_qubits = openfermion.count_qubits(h_ferm) # no. of qubits

# generating results for qubit based partitioning methods (options: fc,fc_si,qwc,qwc_si)
# change method and molecule as needed
trotter(method,mol, h_jw, n_qubits,constant,t)
