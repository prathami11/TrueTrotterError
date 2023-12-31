
To generate the $\epsilon$ table for smaller molecules like H2 and LiH:
1. Directly run the python code trotter_ferm.py for the fermionic partitioning methods and run trotter_qubit.py for the qubit-based partitioning techniques


  To generate the $\epsilon$ table for larger molecules with a high number of fragments for trotterization like, $H_{2}O$, $BeH_{2}$, $NH_{3}$, follow the subsequent steps:


A. **For fermionic partitioning methods**:

  1. Run the code shrink_ham.py to generate the symmetry-adapted Hamiltonians, wavefunctions and the unitary matrices needed to shrink the Hamiltonian fragments. The output will be saved in the "SymFrags" directory.
  2. Then, run the code shrink_fargs.py to generate and store the symmetry-adapted Hamiltonian fragments in their matrix form in the "MatrixFrags" directory.
  3. Finally run trotter_symred.py to import the previously generated symmetry-adapted fragments, wavefunctions and Hamiltonians stored in the "MatrixFrags" and "SymFrags" folder and generate $\epsilon$ values.



B. **For qubit-based partitioning methods**:
   1. For $NH_{3}$ , qubit-tapering has been performed. Run nh3_qubit.py to generate the $\epsilon$ values for the qubit-based partitioning techniques which imports the qubit-tapered fragments from the "Tapered" folder.
   2. Rest of the molecules don't need qubit-tapering, and the $\epsilon$ values can be directly generated from trotter_qubit.py.

To generate the $\alpha_{exact}$ Table for fermionic and qubit-based partitioning techniques, run the codes exact_alphas_ferm.py and exact_alphas_qubit.py respectively.

**Generation of qubit-tapered fragments**

We can use TapQubFrags.py script to perform tapering fragment-wise. It is assumed that the Hamiltonian fragments are pre-computed.
It provides the option of saving the Clifford unitary that renders the Hamiltonian and Hamiltonian fragments in
taperable form. If this option is not chosen, it is assumed that the Clifford unitary was pre-computed and it will be
loaded.
usage: python TapQubFrags.py (mol) (nqubs) (eta) (meth) (encod) (saveCliff)

mol is the name of the molecule
nqubs is the number of qubits that encode the electronic Hamiltonian
eta is the number of electrons of the electronic ground state,
encond can be either bk or jw, depending of the encoding of the Hamiltonian and Hamiltonian fragments
saveCliff is either False or True; if True, the Clifford unitary and eigenvalues to perform tapering are calculated and saved,
otherwise, it is assumed that those were precomputed and saved and  will be loaded.
