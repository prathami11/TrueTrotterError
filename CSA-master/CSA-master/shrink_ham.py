import numpy as np
import openfermion as of
path_prefix="../CSA-master/" #path to CSA-master directory
import sys
sys.path.append(path_prefix)
import saveload_utils as sl #From CSA library
from scipy import sparse
import os
import pickle
mol = "h2" # options: h2, lih, beh2, h2o, nh3
n_elec = 2

h_ferm=sl.load_fermionic_hamiltonian(mol,path_prefix) #loads the Openfermion Fermion operator corresponding to the
                                                      #molecule mol
n_qubits = of.count_qubits(h_ferm) # no. of qubits
Sz_op = of.linalg.get_sparse_operator(of.hamiltonians.sz_operator(n_qubits//2))
Sx_op = of.linalg.get_sparse_operator(of.hamiltonians.sx_operator(n_qubits//2))
Sy_op = of.linalg.get_sparse_operator(of.hamiltonians.sy_operator(n_qubits//2))
# S squared operator
S_sq = Sx_op**2+Sy_op**2+Sz_op**2
n_op = of.linalg.get_sparse_operator(of.hamiltonians.number_operator(n_qubits),n_qubits)
print("exponentiating")
import scipy as sp
#n_elec = 2 # no. of ground state electrons of the respective molecule, modify accordingly
fn =sp.sparse.linalg.expm(-1e8*((n_op-n_elec*sp.sparse.identity(2**n_qubits))**2))
fsq =sp.sparse.linalg.expm(-1e8*((S_sq)**2))
fz = sp.sparse.linalg.expm(-1e8*((Sz_op)**2))
G = fz + fsq + fn
print("diagonalizing")
w, v =np.linalg.eigh(G.todense())
idx = w.argsort()[::-1]
w = w[idx]
#Creating Directory for the molecule to store results, uncomment when needed
#if not os.path.isdir('./SymFrags/'):
#    os.mkdir('./SymFrags/')
np.save('./SymFrags/'+mol+'_w', w)
v = v[:,idx]
np.save('./SymFrags/'+mol+'_v', v) # save unitary matrix
print("starting counter")
counter = 0
for i in range(len(w)):
    if 2.99 <= w[i]<= 3.01:
        print((w[i]),i)
        counter += 1
print(counter)
np.save('./SymFrags/'+mol+'_counter', counter)
vdag=np.conjugate(np.transpose(v))
np.save('./SymFrags/'+mol+'_vdag', vdag)
HamSpOp = of.linalg.get_sparse_operator(h_ferm)
SymHam=np.matmul(vdag,np.matmul(HamSpOp.todense(),v))
SymHam = SymHam[0:counter,0:counter]
np.save('./SymFrags/'+mol+'_SymHam', SymHam) # save symmetry-adapted Hamiltonian
print("Shape of symmetric Hamiltonian:",SymHam.shape)
ExactEigs=of.eigenspectrum(h_ferm)
np.save('./SymFrags/'+mol+'_gs', min(ExactEigs))
TestEigs=np.linalg.eigvalsh(SymHam)
print(min(ExactEigs))
print(min(TestEigs))
hf = of.jw_hartree_fock_state(n_elec,n_qubits) # corresponding to the molecule, ex. beh2
SymHf =np.dot(vdag,hf)
SymHf =np.copy(SymHf[0,0:counter])
np.save('./SymFrags/'+mol+'_SymHf', SymHf) # save symmtery-adapted wavefunction
print(SymHf.shape)
