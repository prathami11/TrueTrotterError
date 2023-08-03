import numpy as np
import openfermion as of
path_prefix="../CSA-master/" #path to CSA-master directory
import sys
sys.path.append(path_prefix)
import saveload_utils as sl #From CSA library
from scipy import sparse
import os
import pickle
h_ferm=sl.load_fermionic_hamiltonian('beh2',path_prefix) #loads the Openfermion Fermion operator corresponding to the
                                                      #molecule mol

Sz_op = of.linalg.get_sparse_operator(of.hamiltonians.sz_operator(7))
Sx_op = of.linalg.get_sparse_operator(of.hamiltonians.sx_operator(7))
Sy_op = of.linalg.get_sparse_operator(of.hamiltonians.sy_operator(7))
# S squared operator
S_sq = Sx_op**2+Sy_op**2+Sz_op**2
n_op = of.linalg.get_sparse_operator(of.hamiltonians.number_operator(14),14) 
print("exponentiating")
import scipy as sp
fn =sp.sparse.linalg.expm(-1e8*((n_op-6*sp.sparse.identity(2**14))**2))
fsq =sp.sparse.linalg.expm(-1e8*((S_sq)**2))
fz = sp.sparse.linalg.expm(-1e8*((Sz_op)**2))
G = fz + fsq + fn
print("diagonalizing")
w, v =np.linalg.eigh(G.todense())
idx = w.argsort()[::-1]   
w = w[idx]
if not os.path.isdir('./SymFrags/'):
    os.mkdir('./SymFrags/')
np.save("./SymFrags/BeH2_w", w)
v = v[:,idx]
np.save("./SymFrags/BeH2_v", v) # save unitary matrix 
print("starting counter")
counter = 0
for i in range(len(w)):
    if 2.99 <= w[i]<= 3.01:
        print((w[i]),i)
        counter += 1
print(counter)
np.save("./SymFrags/BeH2_counter", counter)
vdag=np.conjugate(np.transpose(v))
np.save("./SymFrags/BeH2_vdag", vdag)
HamSpOp = of.linalg.get_sparse_operator(h_ferm)
SymHam=np.matmul(vdag,np.matmul(HamSpOp.todense(),v))
SymHam = SymHam[0:counter,0:counter]
np.save("./SymFrags/BeH2_SymHam", SymHam) # save symmetry-adapted Hamiltonian
print(SymHam.shape)
ExactEigs=of.eigenspectrum(h_ferm)
np.save("./SymFrags/BeH2_gs", min(ExactEigs))
TestEigs=np.linalg.eigvalsh(SymHam)
print(min(ExactEigs))
print(min(TestEigs))
hf = openfermion.jw_hartree_fock_state(6,14) # corresponding to the molecule, ex. beh2
SymHf =np.dot(vdag,hf)
SymHf =np.copy(SymHf[0,0:counter])
np.save("./SymFrags/BeH2_SymHf", SymHf) # save symmtery-adapted wavefunction
print(SymHf.shape)






