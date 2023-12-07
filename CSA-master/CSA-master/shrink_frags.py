import numpy as np
import os
import scipy as sp
import openfermion
path_prefix="../CSA-master/" ##Top of the project, corresponding to CSA/
import sys
sys.path.append(path_prefix)
import time
import pickle
import saveload_utils as sl

def shrink_frags(method, mol, index,n_qubits):
    start = time.time()
    print("Start")
    print("Running for method:",method)
    # Importing Fragments
    f = open("./Frag_Lib/"+method+"/"+mol+"_"+method+"Frags", 'rb')
    dict = pickle.load(f)
    f.close()
    ListFrags=dict['grouping'] # stores all the fragments
    number_frags = len(ListFrags) # Number of fragments
    print(number_frags)
    for i in range(len(ListFrags)):
        print(i)
        ListFrags[i] = openfermion.linalg.get_sparse_operator(ListFrags[i],n_qubits)
    v = np.load('./SymFrags/'+mol+'_v.npy') # load the unitary matrix for the respective molecule
    vdag = np.load('./SymFrags/'+mol+'_vdag.npy')
    print("Shrinking")
    for i in range(len(ListFrags)):
        print(i)
        x = np.matmul(vdag,np.matmul(ListFrags[i].todense(),v))
        np.save("./MatrixFrags/"+mol+"/"+method+"/"+str(i), x[0:index,0:index])# change index value wrt the molecule
    total_time = time.time()-start
    print("time in hrs",total_time/3600)
index = 3 # options 3 for h2, 105 for lih, 490 for beh2, 196 for h2o, 8008 for nh3
mol = "h2" # options: h2, lih, beh2, h2o, nh3
h_ferm=sl.load_fermionic_hamiltonian(mol,path_prefix) #
n_qubits = openfermion.count_qubits(h_ferm) # no. of qubits
method = "FRO" #options: FRO, GFRO, GFROLCU, LRLCU, svd, GCSASD
shrink_frags(method,mol,index,n_qubits) #specify the method
