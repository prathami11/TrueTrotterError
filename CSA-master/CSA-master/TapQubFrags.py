"""
Script to perform tapering fragment-wise. It is assumed that the Hamiltonian fragments are pre-computed.
It provides the option of saving the Clifford unitary that renders theHamiltonian and Hamiltonian fragment in
taperable form. If this option is not chosen, it is assumed that the Clifford unitary was pre-computed and it will be
loaded.
usage: python TapQubFrags.py (mol) (nqubs) (eta) (meth) (encod) (saveCliff)

eta is the number of electrons of the electronic ground state,
encond can be either bk or jw, depending of the encoding of the Hamiltonian and Hamiltonian fragments
saveCliff is either False or True; if True, the Clifford unitary and eigenvalues to perform tapering are calculated and saved,
otherwise, it is assumed that those were precomputed and saved and  will be loaded.
"""
import openfermion
import numpy as np
import pickle
import os
import sys
import h5py
sys.path.append('./utils')

import tapering_utils as taps

path_prefix="./"
#path_prefix='../../../Trotterization/CSA-master/'
sys.path.append(path_prefix)

import saveload_utils as sl #From CSA library

import ferm_utils
#from qubit_utils import get_qwc_group,get_greedy_grouping
#from trotint import *
from scipy import sparse

Root_Frags='./Frag_Lib/'

def SaveFrag(FileName,method,SpFrag,i):

    if os.path.exists(FileName):
        f = h5py.File(FileName,'a')
        g= f[method]

    else:
        f = h5py.File(FileName,'w')
        g = f.create_group(method)
        #print(f"The file '{file_path}' does not exist.")

    g.create_dataset('data'+str(i),data=SpFrag.data)
    g.create_dataset('indptr'+str(i),data=SpFrag.indptr)
    g.create_dataset('indices'+str(i),data=SpFrag.indices)
    g.create_dataset('shape'+str(i),data=SpFrag.shape)

    f.close()
    return


#definition of Fock operator...
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

def ReadBool(String):
    if String=='False' or String=='false':
        return False
    elif String=='True' or String=='true':
        return True
    else:
        print("Input not recognized, assume it is True")
        return True

#Doing tapering in the original basis...
#mol='h2'
#nqubs=4
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
nqubs = 4 if len(sys.argv) < 3 else int(sys.argv[2])
eta = 2 if len(sys.argv) < 4 else int(sys.argv[3])
meth = 'qwc' if len(sys.argv) < 5 else sys.argv[4]
encod = 'jw' if len(sys.argv) < 6 else sys.argv[5]

saveCliff = True if len(sys.argv) < 7 else ReadBool(sys.argv[6])

#Save to be read to retrieve fragments...
FileName=Root_Frags+meth+'/'+mol+'_'+meth+'Frags'

SaveFol='./TapFrags/'+mol+'/'+meth+'/'
SaveCliffFol='./SymRed/'

if saveCliff:

    h_orig=sl.load_fermionic_hamiltonian(mol,prefix=path_prefix)

    FockOp=GetFockOp(h_orig,eta)

    orig_shift=h_orig.constant

    if encod == 'bk':
        h_bksum = openfermion.bravyi_kitaev(h_orig+FockOp)
        h_bkconst=h_bksum.constant
        h_bk = openfermion.bravyi_kitaev(h_orig)
    elif encod == 'jw':
        h_bksum = openfermion.jordan_wigner(h_orig+FockOp)
        h_bkconst=h_bksum.constant
        h_bk = openfermion.jordan_wigner(h_orig)

    h_bksum_noconstant = h_bksum - h_bksum.constant; h_bksum_noconstant.compress()
    #h_bk_noconstant = h_bk

    SymOps,ComQub,ComHam=taps.GenQubitSym(h_bksum_noconstant)
    CliffUnit=taps.GenCliffUnit(h_bksum)

###Rotate the original Hamiltonian...

#if saveCliff:
    RotHam=openfermion.hermitian_conjugated(CliffUnit)*h_bk*CliffUnit

    DRotSyms=[]
    for i in range(len(SymOps)):
        Dum=openfermion.hermitian_conjugated(CliffUnit)*SymOps[i]*CliffUnit
        DRotSyms.append(Dum)

    RotSyms=[]
    for i in range(len(DRotSyms)):
        for j in DRotSyms[i].terms:
            coeff=DRotSyms[i].terms[j]
            #print(coeff)
            if np.abs(coeff)>1e-4:
                RotSyms.append(openfermion.QubitOperator(j))

    RedSymList=[]
    for i in range(len(RotSyms)):
        for j in RotSyms[i].terms:
            #print(len(j))
            if len(j)==1:
                RedSymList.append(RotSyms[i])

    EigNums=np.zeros(len(RedSymList))
    SpHBK=openfermion.get_sparse_operator(RotHam)
    print("Starting calculation of ground state of rotated Hamiltonian...")
    eigBKval,eigBKvec=sparse.linalg.eigsh(SpHBK,k=1,which='SA')
    for i in range(len(RedSymList)):
        SpSym=openfermion.get_sparse_operator(RedSymList[i],n_qubits=nqubs)

        EigNums[i]=round(np.dot(np.conjugate(np.transpose(eigBKvec[:,0])),SpSym*eigBKvec[:,0]))

    print("Tapering Hamiltonian...")
    TappHam=taps.TapperRotHam(EigNums,RedSymList,RotHam,nqubs)
    #Verify the integrity of the Tappered Hamiltonian

    if mol=='h2':
        EigsOr=openfermion.eigenspectrum(h_orig)
        EigsTap=openfermion.eigenspectrum(TappHam)
        GeTap=np.min(EigsTap)
        GeOr=np.min(EigsOr)
    else:

        SpTapHam=openfermion.get_sparse_operator(TappHam)
        SpOrigHam=openfermion.get_sparse_operator(h_orig)

        eigTap,eigvTap=sparse.linalg.eigsh(SpTapHam,k=1,which='SA')
        GeTap=eigTap[0]

        eigOr,eigvOr=sparse.linalg.eigsh(SpOrigHam,k=1,which='SA')
        GeOr=eigOr[0]
    print("Difference between ground state of tapered and original Hamiltonians:", GeOr-GeTap)
    SaveCliffDir=SaveCliffFol+mol+'/'

    if not os.path.exists(SaveCliffDir):
        os.makedirs(SaveCliffDir)


    SaveClifFile=SaveCliffDir+mol+'Cliff.pk'
    SDict={}
    SDict['CliffUnit']=CliffUnit
    SDict['RotSyms']=RedSymList
    SDict['EigSyms']=EigNums
    f=open(SaveClifFile,'wb')
    pickle.dump(SDict,f)
    f.close()

else:
    LoadFil=SaveCliffFol+mol+'/'+mol+'Cliff.pk'
    f=open(LoadFil,'rb')
    dat=pickle.load(f)
    f.close()
    CliffUnit=dat['CliffUnit']
    RedSymList=dat['RotSyms']
    EigNums=dat['EigSyms']

#Perform tappering of each of the Hamiltonian fragments...
f=open(FileName,'rb')
dat=pickle.load(f)

Gps=dat['grouping']


ListTap=[]

if not os.path.exists(SaveFol):
    os.makedirs(SaveFol)

savefilname=SaveFol+mol+'_'+meth+'_SymFrags.h5'

RemQubs=nqubs-len(RedSymList)
for i in range(len(Gps)):
    #SpFrag=openfermion.get_sparse_operator(dat['grouping'][i],n_qubits=nqubs)
    Frag=Gps[i]
    RotFrag=openfermion.hermitian_conjugated(CliffUnit)*Frag*CliffUnit
    TapFrag=taps.TapperRotHam(EigNums,RedSymList,RotFrag,nqubs)
    #Save in sparse form and hdf5 format....
    SaveFrag(savefilname,meth,openfermion.get_sparse_operator(TapFrag,n_qubits=RemQubs),i)
    ListTap.append(TapFrag)


if not os.path.exists(SaveFol):
    os.makedirs(SaveFol)

Dict={}
Dict['grouping']=ListTap
Dict['encoding']=encod
#Dict['n_qubits']=openfermion.count_qubits(TappHam) #We save the number of qubits after tapering!!!
Dict['RotSyms']=RedSymList # the list of symmetries that are applied for tapering after clifford rotation
#Dict['NPwords']=Npauls-1 #minus one, to account for the constant in TappHam
Dict['CliffUnit']=CliffUnit
Dict['EigSyms']=EigNums


savefilname=SaveFol+mol+'_'+meth+'_SymFrags.pk'
f=open(savefilname,'wb')
pickle.dump(Dict,f)
f.close()
