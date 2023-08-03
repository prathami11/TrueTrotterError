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

def shrink_frags(method):
    start = time.time()
    print("Start")
   
    print("Running for method:",method)
    # Importing Fragments
    f = open("./Frag_Lib/"+method+"/beh2_"+method+"Frags", 'rb')
    dict = pickle.load(f)
    f.close()
    ListFrags=dict # stores all the fragments
    number_frags = len(ListFrags) # Number of fragments
    print(number_frags)
    for i in range(len(ListFrags)):
        print(i)
        ListFrags[i] = openfermion.linalg.get_sparse_operator(ListFrags[i],14)
    v = np.load("./SymFrags/BeH2_v.npy") # load the unitary matrix 
    vdag = np.load("./SymFrags/BeH2_vdag.npy")
    print("Shrinking")
    for i in range(len(ListFrags)):
        print(i)
        x = np.matmul(vdag,np.matmul(ListFrags[i].todense(),v))
        np.save("./MatrixFrags/BeH2/GFRO/"+str(i), x[0:490,0:490])# change index value wrt the molecule
    total_time = time.time()-start
    print("time in hrs",total_time/3600)

shrink_frags("GFROLCU") #specify the method


