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

def trotter(method, h_jw, n_qubits,constant,t):
    start = time.time()
    print("Start")
    #if not os.path.isdir('./New_Results/H2'):
    #    os.mkdir('./New_Results/H2/')
    # Importing Fragments
    f = open("./Frag_Lib/"+method+"/h2_"+method+"Frags", 'rb')
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
    print(z)
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
    f=open('./Exact_Alphas/H2/'+method,'wb')
    pickle.dump(results,f)
    f.close()
    print("Done executing after",total_time, "hrs")

# hamiltonian
h_ferm=sl.load_fermionic_hamiltonian("h2",path_prefix) # fermionic hamiltonian wrt molecule
h_jw = openfermion.transforms.jordan_wigner(h_ferm)
constant = h_jw.constant
n_qubits = openfermion.count_qubits(h_ferm) # no. of qubits
t=1e-5 # time step
# generating results for qubit based partitioning methods (options: fc,fc_si,qwc,qwc_si)
trotter("fc", h_jw, n_qubits,constant,t)
trotter("fc_si",h_jw, n_qubits,constant,t)
trotter("qwc",h_jw, n_qubits,constant,t)
trotter("qwc_si",h_jw,n_qubits,constant,t)