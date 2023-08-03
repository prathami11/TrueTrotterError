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
from scipy.sparse import linalg
import pickle
import saveload_utils as sl
import time

def trotter(method, n_qubits, gs, constant):
    start = time.time()
    print("Start")
    if not os.path.isdir('./Results/BeH2'):
        os.mkdir('./Results/BeH2/')
    # openfermion hartree fock wavefunction in jordan wigner representation
    print("Running for method:",method)
    hf = openfermion.jw_hartree_fock_state(6,14)

    # Importing Fragments
    f = open("./Frag_Lib/"+method+"/beh2_"+method+"Frags", 'rb')
    dict = pickle.load(f)
    f.close()
    ListFrags=dict['grouping'] # stores all the fragments
    number_frags = len(ListFrags) # Number of fragments
    ListFrags[0] += openfermion.QubitOperator("",constant ) # adding the constant term
    #get the sparse matrix representation of the Fragments.
    jw_ham_sp=[]
    for i in range(len(ListFrags)):
        jw_ham_sp.append(openfermion.linalg.get_sparse_operator(ListFrags[i],n_qubits))
    
    ## Defining control parameters
    # Total Time of propagation
    T = 10000000
    # Sample Spacing i.e, T/deltaT is the No. of points fed to the fourier tranform
    deltaT = 0.2
    m = np.arange(1,11,1)
    # trotter steps
    deltat = deltaT/m
    # discrete evolution times
    t = np.arange(0,T,deltaT)

    # Changes in Ground state energy due to trotterization for different trotter steps
    trotter_error = [] # stores the error in the ground state energy value due to trotterization for different trotter steps
    for k in range(len(deltat)):
        
        print("Running for data point",k)
        # trotter number 
        n = np.arange(0,T/deltat[k],deltaT/deltat[k])
        n = n.astype(int)
        # constructing propagator dt
        trotter_state = linalg.expm(-1j*jw_ham_sp[0]*deltat[k])
        for j in range(len(jw_ham_sp)-1):
            trotter_state = linalg.expm_multiply(-1j*jw_ham_sp[j+1]*deltat[k],trotter_state)
        matrix = trotter_state.todense() 
        # diagonalizaion of the propagator with eigendecomposition
        w, v = sp.linalg.eig(matrix)
        u = np.stack(v)
        u_dag_sp = sp.sparse.csr_matrix((u.conj()).T)
        phi = u_dag_sp@hf
        c= np.ravel((np.multiply(phi.conj(),phi)).T)
        new_c = [] # storing the significant elements of c_sq vector
        d = [] # storing the corresponding elements of the D vector
        for i in range((c).shape[0]):
            if  c[i] >= 1e-6:  # filtering out the significant terms
                new_c.append(c[i])
                d.append(w[i])
        product_state = []  # propagator(t_i)
        trotter_expectation = [] # autocorrelation fucntion
        product_state.append(d**n[0])
        trotter_expectation.append(np.inner(new_c,product_state[0]))
        for i in range(len(n)-1):
            product_state.append(np.multiply(product_state[i],(d)**(n[i+1]-n[i])))
            trotter_expectation.append(np.inner(new_c,product_state[i+1]))
        # fourier transform
        xf_trotter = 2*np.pi*fftfreq(len(t),deltaT)
        yf_trotter = ifft(np.array(trotter_expectation))   
        peaks_trotter, _ = find_peaks(np.abs(yf_trotter))
        results_half_trotter = peak_widths(np.abs(yf_trotter), peaks_trotter, rel_height=0.5)
        trotter_error.append( min(xf_trotter[peaks_trotter])-gs)
    results = {}
    total_time = time.time()-start
    total_time = total_time/3600 # in hrs
    dt_sq = deltat**2
    #slope, intercept, r_value, p_value, std_err = sp.stats.linregress(dt_sq, trotter_error)
    #results['slope'] = slope
    results['dt_sq'] = dt_sq
    #results['r_value']= r_value
    results['trotter_error']= trotter_error
    results['number_frags'] = number_frags
    results['total_time'] = total_time
    f=open('./Results/BeH2/'+method,'wb')
    pickle.dump(results,f)
    f.close()
    print("Done executing after",total_time, "hrs")
#if not os.path.isdir('./Results/'):
#    os.mkdir('./Results/')
h_ferm=sl.load_fermionic_hamiltonian("beh2",path_prefix) # fermionic hamiltonian
n_qubits = openfermion.count_qubits(h_ferm) # no. of qubits
h_jw = openfermion.transforms.jordan_wigner(h_ferm) # jordan wigner transformatio for qubit operator
# eigenspectrum of the hamiltonian
spectrum = openfermion.eigenspectrum(h_jw)
gs = min(spectrum) # ground state energy
constant = h_jw.constant 
# generating results for qubit based partitioning methods (options: fc, qwc, fc_si, qwc_si)
# coment / uncomment these lines for the required methods
trotter("fc",n_qubits,gs,constant)
trotter("fc_si",n_qubits,gs,constant)
trotter("qwc", n_qubits,gs,constant)
trotter("qwc_si",n_qubits,gs,constant)
