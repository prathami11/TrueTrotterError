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

def trotter(method,mol,n_qubits,gs,deltaT):
    
    #Creating Directory for the molecule to store results, uncomment when needed
    #if not os.path.isdir('./Results/'+mol):
    #    os.mkdir('./Results/'+mol+'/')
    print("Running for method:",method)
    # openfermion hartree fock wavefunction in jordan wigner representation
    hf = openfermion.jw_hartree_fock_state(6,14) #change with repsective to molecule 
    # Importing Fragments
    f = open("./Frag_Lib/"+method+"/"+mol+"_"+method+"Frags", 'rb')
    dict = pickle.load(f)
    f.close()
    ListFrags=dict['grouping'] # stores all the fragments
    number_frags = len(ListFrags) # Number of fragments
    # Map the Fragments to QubitOperator using the JWT
    jw_ham = []
    from openfermion import jordan_wigner
    for i in range(len(ListFrags)):
        jw_ham.append(jordan_wigner(ListFrags[i]))

    #get the sparse matrix representation of the Fragments.
    jw_ham_sp=[]
    for i in range(len(ListFrags)):
        jw_ham_sp.append(openfermion.linalg.get_sparse_operator(jw_ham[i],n_qubits))
    
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
    dt_sq = deltat**2
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(dt_sq, trotter_error)
    results['slope'] = slope
    results['dt_sq'] = dt_sq
    results['r_value']= r_value
    results['trotter_error']= trotter_error
    results['number_frags'] = number_frags
    f=open('./Results/'+mol+'/'+method,'wb')
    pickle.dump(results,f)
    f.close()
# will generate an error if directory already exists
#if not os.path.isdir('./Results/'):
#    os.mkdir('./Results/')
# importing femrionic hamiltonian
mol = "beh2" # options: h2, lih, beh2, h2o, nh3
method = "svd" # options: svd, SVDLCU, FRO, GFRO, GFROLCU, GCSASD
h_ferm=sl.load_fermionic_hamiltonian(mol,path_prefix) # fermionic hamiltonian
n_qubits = openfermion.count_qubits(h_ferm) # no. of qubits
h_jw = openfermion.transforms.jordan_wigner(h_ferm) # jordan wigner transformatio for qubit operator
# eigenspectrum of the hamiltonian
spectrum = openfermion.eigenspectrum(h_jw)
gs = min(spectrum) # ground state energy
# Sample Spacing i.e, T/deltaT is the No. of points fed to the fourier tranform
deltaT = 0.2 # change with molecule # 2 for h2, 0.39 for lih, 0.2 for beh2, 0.04 for h2o, 0.05 for nh3 
# generating results for fermionic based partitioning methods 
trotter(method,mol, n_qubits, gs,deltaT)






    
