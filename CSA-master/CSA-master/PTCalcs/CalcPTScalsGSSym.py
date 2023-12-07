#Usage: python CalcPTScalsGSSym.py (mol) (nel) (meth) (nprocs) (C2Err) (C1Err) (restartC2) (Nsegs)
#To implement restart capability...Nsegs is the number of chunks that will be processed in parallel
#after the end of each processed chunk, it will be saved in a file. The results will be incremented
#segment by segment.

import openfermion
import numpy as np
import pickle
from joblib import Parallel, delayed
import sys
import os
path_prefix='../'
sys.path.append(path_prefix)

import saveload_utils as sl #From CSA library
import ferm_utils
from scipy import sparse
import h5py
import scipy as sp

def count_npy_files(directory):
    npy_count = 0

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file has a .npy extension
            if file.endswith(".npy"):
                npy_count += 1

    return npy_count

def LoadAllFrags(mol,meth,rootDir):

    topPath=rootDir+mol+'/'+meth+'/'

    NFrags=count_npy_files(topPath)

    Frags=[]
    for i in range(NFrags):
        Frags.append(sparse.csc_matrix(np.load(topPath+str(i)+'.npy')))

    return Frags


def loadFrag(FileName,meth,i):

    f = h5py.File(FileName, 'r')


    data=f[meth]['data'+str(i)]
    indptr=f[meth]['indptr'+str(i)]
    indices=f[meth]['indices'+str(i)]
    shape=f[meth]['shape'+str(i)]

    csc_mat = sparse.csc_matrix((data, indices, indptr), shape=shape)

    return csc_mat

def LoadAllFragsTap(FileName,meth,Nfrags):

    Frags=[]
    for i in range(Nfrags):
        Frags.append(loadFrag(FileName,meth,i))

    return Frags


def GetFockOp(h_ferm,nel):
    '''
    Obtains a fock operator, whose spectrum and eigenstates can be used in our perturbative
    calculations...
    '''
    Norbs=openfermion.count_qubits(h_ferm)
    VecHF=openfermion.jw_hartree_fock_state(nel,Norbs)
    NspOrb=Norbs//2

    OneBodFock=np.zeros([NspOrb,NspOrb])

    h_ferm_sp=openfermion.get_sparse_operator(h_ferm)

    for p in range(NspOrb):
        for q in range(NspOrb):

            for alpha in range(2):
                spin_p=2*p+alpha
                spin_q=2*q+alpha

                #OpDum=openfermion.commutator(openfermion.FermionOperator(str(spin_p)+'^ '+str(spin_q)),h_ferm)
                aqdag_o=openfermion.FermionOperator(str(spin_q)+'^ ')
                ap_o=openfermion.FermionOperator(str(spin_p))

                aqdag_osp=openfermion.get_sparse_operator(aqdag_o,n_qubits=Norbs)
                ap_osp=openfermion.get_sparse_operator(ap_o,n_qubits=Norbs)

                #OpDum=openfermion.FermionOperator(str(spin_q)+'^ ')*openfermion.commutator(openfermion.FermionOperator(str(spin_p)),h_ferm)
                #OpDum+=openfermion.commutator(openfermion.FermionOperator(str(spin_p)),h_ferm)*openfermion.FermionOperator(str(spin_q)+'^ ')
                Comm=ap_osp*h_ferm_sp-h_ferm_sp*ap_osp
                OpDum_sp=aqdag_osp*Comm
                OpDum_sp+=Comm*aqdag_osp

                #OpDum_sp=openfermion.get_sparse_operator(OpDum)

                expect=np.dot(VecHF,OpDum_sp*VecHF)

                #print(expect)

                OneBodFock[p,q]+=expect

    OneBodFock=0.5*OneBodFock
    return ferm_utils.get_ferm_op(OneBodFock)


def krondelt(i,j):
    if i==j:
        return 1.0
    else:
        return 0.0
def commut(Op1,Op2):

    return Op1*Op2-Op2*Op1

def ReadBool(String):
    if String=='False' or String=='false':
        return False
    elif String=='True' or String=='true':
        return True
    else:
        print("Input not recognized, assume it is True")
        return True


RootSaveFold='./PTResults/'

pathSymFrags='../MatrixFrags/'
pathTapFrags='../TapFrags/'

mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
nel = 2 if len(sys.argv) < 3 else int(sys.argv[2])
meth= 'FRO' if len(sys.argv) < 4 else sys.argv[3]
nprocs= 2 if len(sys.argv) < 5 else int(sys.argv[4])
C2Err =True if len(sys.argv) < 6 else ReadBool(sys.argv[5])
C1Err=True if len(sys.argv) < 7 else ReadBool(sys.argv[6])
restartC2=False if len(sys.argv) < 8 else ReadBool(sys.argv[7])
Nsegs=10 if len(sys.argv) < 9 else int(sys.argv[8])

logpath=RootSaveFold+mol+'/'+meth+'/'
if not os.path.exists(logpath):
    os.makedirs(logpath)

log_file = open(logpath+'logfile.txt', 'w')
sys.stdout = log_file

print("Molecule is: ",mol)
print("Method is: ",meth)
print("Number of electrons: ",nel)

listQubMets=['qwc','qwc_jw','fc','fc_jw','qwc_si','qwc_si_jw','fc_si','fc_si_jw']


SaveEigs=True

if meth in listQubMets:
    FileName=pathTapFrags+mol+'/'+meth+'/'+mol+'_'+meth+'_SymFrags.h5'
    f=h5py.File(FileName,'r')

    NFrags=int(len(f[meth].keys())/4)
    f.close()

    Frags=LoadAllFragsTap(FileName,meth,NFrags)
else:
    Frags=LoadAllFrags(mol,meth,pathSymFrags)
    NFrags=len(Frags)


OneBod_sp=Frags[0]
for i in range(1,len(Frags)):
    OneBod_sp+=Frags[i]

if NFrags>20:
    #Nsegs=20
    #Nsegs=2
    SizeSeg=int(np.floor(NFrags/Nsegs))

else:
    Nsegs=1
    SizeSeg=NFrags

if NFrags%Nsegs:
    Nsegs+=1


print("NFrags is",NFrags)

if C2Err:
    print("Entering C2Err")
    print("Nsegs is:",Nsegs)
    print("SizeSeg is:",SizeSeg)
    #GE,GS=sparse.linalg.eigsh(h_ferm_sp,k=1,which='SA')
    GE,GS=sparse.linalg.eigsh(OneBod_sp,k=1,which='SA')

    def process_fragment(mu,Frags=Frags,NFrags=NFrags):
        #Hmu_sp = openfermion.get_sparse_operator(Frags[mu], n_qubits=nqubs)
        Hmu_sp = Frags[mu]
        shape = Hmu_sp.shape

        #C1Op_local = 0.0
        C2Op_local = sparse.csc_matrix(np.zeros(shape))

        for v in range(mu+1, NFrags):
#            Hv_sp = openfermion.get_sparse_operator(Frags[v], n_qubits=nqubs)
            Hv_sp = Frags[v]

            #C1Op_local += commut(Hmu_sp, Hv_sp)

            for vprim in range(v, NFrags):
                #Hvprim_sp = openfermion.get_sparse_operator(Frags[vprim], n_qubits=nqubs)
                Hvprim_sp = Frags[vprim]

                Ops = commut(Hvprim_sp, commut(Hv_sp, Hmu_sp))

                C2Op_local += -(1.0 - 0.5 * krondelt(vprim, v)) * Ops

        return C2Op_local#, C1Op_local,

    ####Phase of operator-computation ####
    savepath=RootSaveFold+mol+"/"+meth+"/"
    savename=mol+"_"+meth+"C2OPGS.pk"

    if restartC2:
        #load the accumlated operator...
        file_path_name=savepath+savename
        with open(file_path_name,'rb') as handle:
            LoadDict=pickle.load(handle)

        LoadC2Op=LoadDict['C2Op']
        Iter=LoadDict['Iter']
        Init=Iter
    else:
        LoadDict={}
        LoadC2Op=sparse.csc_matrix(np.zeros_like(OneBod_sp.toarray()))
        Iter=0
        Init=0

    for i in range(Init,Nsegs):
        print("Entering loop")

        if i==(Nsegs-1):
            results = Parallel(n_jobs=nprocs)(
                delayed(process_fragment)(mu) for mu in range(i*SizeSeg,NFrags)
            )
        else:
            results = Parallel(n_jobs=nprocs)(
                delayed(process_fragment)(mu) for mu in range(i*SizeSeg,(i+1)*SizeSeg)
            )


        C2Op = sum(r for r in results)
        print("Computation of C2 operator for iteration:",i)
        print("Norm of C2Op in this iteration is:",np.linalg.norm(C2Op.toarray(),ord=2))
        savepath=RootSaveFold+mol+"/"+meth+"/"
        savename=mol+"_"+meth+"C2OPGSSym.pk"
        if not os.path.exists(savepath):
            os.makedirs(savepath)


        LoadC2Op+=C2Op
        Iter+=1

        LoadDict['C2Op']=np.copy(LoadC2Op)
        LoadDict['Iter']=Iter

        file_path_name=savepath+savename
        with open(file_path_name,'wb') as handle:
            pickle.dump(LoadDict,handle,protocol=pickle.HIGHEST_PROTOCOL)


    #Calculating first order scaling PT to energy

    E1_GS=np.dot(np.conj(np.transpose(GS)),np.dot(LoadC2Op.toarray(),GS))/3

    savepath=RootSaveFold+mol+"/"+meth+"/"
    savename=mol+"_"+meth+"PT1ScalsParGSSym.pk"

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    #SaveFilName=RootRes+mol+"/"+meth+"/"+mol+"_"+meth+"PTScals.pk"
    DictRes={}
    DictRes['PT1']=E1_GS
    file_path_name=savepath+savename
    with open(file_path_name, 'wb') as handle:
        pickle.dump(DictRes, handle, protocol=pickle.HIGHEST_PROTOCOL)

if C1Err:
    #savepath=RootSaveFold+mol+"/"+meth+"/"
    #savename=mol+"_"+meth+"PT2ScalsPar.pk"

    if not os.path.exists(savepath):
        os.makedirs(savepath)


    def process_fragmentOp1(mu):
        #Hmu_sp = openfermion.get_sparse_operator(Frags[mu], n_qubits=nqubs)
        Hmu_sp = Frags[mu]
        shape = Hmu_sp.shape

        #C1Op_local = 0.0
        C1Op_local = sparse.csc_matrix(np.zeros(shape))
        #C2Op_local = 0.0

        for v in range(mu+1, NFrags):
            #Hv_sp = openfermion.get_sparse_operator(Frags[v], n_qubits=nqubs)
            Hv_sp = Frags[v]

            C1Op_local += commut(Hmu_sp, Hv_sp)

        return C1Op_local


    results = Parallel(n_jobs=nprocs)(
        delayed(process_fragmentOp1)(mu) for mu in range(NFrags)
    )

    #C1Op = sum(r[0] for r in results)
    C1Op = sum(r for r in results)
    print("Computation of C1 operator finished, saving the operator...")
    savepath=RootSaveFold+mol+"/"+meth+"/"
    savename=mol+"_"+meth+"C1OPGSSym.pk"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    DictOp={}
    DictOp['C1Op']=C1Op #in sparse form
    file_path_name=savepath+savename
    with open(file_path_name,'wb') as handle:
        pickle.dump(DictOp,handle,protocol=pickle.HIGHEST_PROTOCOL)

#Calculating the next contribution....
#diagonalize the exact Hamiltonian...

    #Eigs,Eigvects=np.linalg.eigh(h_ferm_sp.toarray())

    if SaveEigs:
        NpzName=RootSaveFold+mol+'/'+mol+'_DiagResHF.npz'
        Eigs,Eigvects=np.linalg.eigh(OneBod_sp.toarray())

        np.savez(NpzName, array1=Eigs, array2=Eigvects)

    else:
        NpzName=RootSaveFold+mol+'/'+mol+'_DiagResGS.npz'
        data = np.load(NpzName)

        # Extract the arrays by name
        Eigs = data['array1']
        Eigvects = data['array2']

    #TODO: save diagonalization results?
    #**********
    def calculate_E2_GS(i):
        num = np.dot(np.conjugate(np.transpose(Eigvects[:, i])), np.dot(C1Op.toarray(), GS))
        den = GE - Eigs[i]
        return 0.25 * np.abs(num)**2 / den

    results = Parallel(n_jobs=nprocs)(
        delayed(calculate_E2_GS)(i) for i in range(1, len(Eigs))
    )

    E2_GS = sum(results)

    #*******


    #saving results...
    savepath=RootSaveFold+mol+"/"+meth+"/"
    savename=mol+"_"+meth+"PT2ScalsParGSSym.pk"

    if not os.path.exists(savepath):
        os.makedirs(savepath)

#SaveFilName=RootRes+mol+"/"+meth+"/"+mol+"_"+meth+"PTScals.pk"
    DictRes={}
    #DictRes['PT1']=E1_GS
    DictRes['PT2']=E2_GS

    file_path_name=savepath+savename
    with open(file_path_name, 'wb') as handle:
        pickle.dump(DictRes, handle, protocol=pickle.HIGHEST_PROTOCOL)

log_file.close()
