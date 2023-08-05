To generate the $\epsilon$ table for the fermionic partitioning methods for smaller molecules like H2 and LiH, directly run the python code trotter_ferm.py and run trotter_qubit.py for the qubit-based partitioning techniques.
Run the code shrink_ham.py to generate the symmetry-adapted Hamiltonians, wavefunctions and the unitary matrices to shrink the fragments. These will be saved in the "SymFrags" directory.
Then Run the code shrink_fargs.py to generate and store the symmetry-adapted Hamiltonian fragments in their matrix form in the MatrixFrags directory.
For larger molecules with a high number of fragments for trotterization, like BeH2, H2O, run trotter_symred.py to use the symmetry-adapted fragments, wavefunctions and Hamiltonians stored in the "MatrixFrags" and "SymFrags" folder and generate $\epsilon$ values.
Run nh3_qubit.py to generate the $\epsilon$ values for the qubit-based partitioning techniques for NH3 which imports the qubit-tapered fragments from the "Tapered" folder.

To generate the $\alpha_{exact}$ Table for fermionic and qubit-based partitioning techniques, run the codes exact_alphas_ferm.py and exact_alphas_qubit.py respectively.
