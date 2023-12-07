Perturbative estimations of the scaling ($\epsilon$) of energy shift, with the square of the Trotter time step, of the ground state energy eigenvalues due to the Trotter approximation
can be computed with these scripts. Given the significant computational overhead in the calculations for systems encoded with more than 10 qubits, all calculations are carried out with symmetry-reduced Hamiltonian fragments that are assumed to be generated and stored in the upper directory.

To obtain the perturbative scalings, our so-called C1 and C2 operators need to be generated in a first stage, which can be accomplished with commands of  the form

python CalcPTScalsGSSym.py (mol) (nel) (meth) (nprocs) (C2Err) (C1Err) (restartC2) (Nsegs)

where mol is either 'h2','lih','beh2','h2o','nh3' is the name of molecule. nel is the number of electrons of the neutral molecule, meth is the name of the partition method, and it must be the name of any of the partition methods whose fragments and symmetry-shrinking were processed in the directory above. nprocs is the number of processes in parallel to perform the calculation, C2Err and C1Err are boolean variables that determine whether the C2 and C1 operators are going to be calculated, respectively. restartC2 is also a boolean variable to restart the calculation of the high-demanding C2 operator. Finally, Nsegs is an integer, such that NFrags/Nsegs is the number of chunks that will be processed in parallel in the calculation of C2 operator.
The default input is mol='h2', nel=2, meth='FRO', nprocs=2, C2Err=True, C1Err=True, restart=False, Nsegs=10.

On top of the calculation of C1 and C2 operators, this script computes the scaling \epsilon on the exact ground-state of the electronic Hamiltonian.

This script generates and stores results under an automatically generated PTResults directory.

To obtain heuristic scalings $\epsilion_{PT}$ based on eigenstates and eigenvalues of a mean-field approximated electronic Hamiltonian, one can use the script PTEsts.py according to the command

python PTEsts.py (mol) (nqubs) (meth) (eta) (states)

where mol is the name of the molecule, nqubs is the number of qubits that encode the electronic Hamiltonian, meth is the name of the partition method, eta is the number of electrons of the neutral molecule, and states can be either 'HF' or 'Exact'. When it is 'HF', the scalings using the fock operator are calculated and stored under PTResults directory. For completeness, it also computes the scalings $\epsilon$ on the exact ground state for states='Exact'
The default input is mol='h2',nqubs=4,meth='qwc',eta=2,states='HF'




