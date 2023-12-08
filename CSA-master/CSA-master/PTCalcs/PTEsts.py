#yusage: python PTEsts.py (mol) (nqubs) (meth) (eta) (states)
#where states can be either 'Exact' or 'HF'. For the former, the calcualtions are done with exact eigenstates
#and eigenspectrum of the Hamiltonian. For the latter, with the spectrum and eigenstates of the Fock op[erator

import openfermion
import numpy as np
path_prefix='../'
import sys
sys.path.append(path_prefix)
sys.path.append(path_prefix+'utils/')

from scipy import sparse
import os

import saveload_utils as sl
import ferm_utils
import pickle
import h5py

import tapering_utils as taps

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


def PT2Est(Eigs,Eigvects,C1Op):
    GE=Eigs[0]
    GS=Eigvects[:,0]

    PTCorr=0.0
    for i in range(1,len(Eigs)):
        num = np.dot(np.conjugate(np.transpose(Eigvects[:, i])), C1Op.dot(GS))
        den = GE - Eigs[i]
        PTCorr+=np.abs(num)**2/den


    return 0.25 * PTCorr

mol='h2' if len(sys.argv) < 2 else sys.argv[1]
nqubs=4 if len(sys.argv) < 3 else int(sys.argv[2])
meth='qwc' if len(sys.argv) < 4 else sys.argv[3]
eta=2 if len(sys.argv) < 5 else int(sys.argv[4])
states='HF' if len(sys.argv) < 6 else sys.argv[5]

#TO change accordingly....
SymFragPath='../SymFrags/'
SymMatrixPath='../MatrixFrags/'
PTRes='./PTResults/'
pathTap='../TapFrags/'

#Dictionary of indexes to perform shrinking of fermionic Hamiltonians...
IdxFerm={}
IdxFerm['h2']=3
IdxFerm['lih']=105
IdxFerm['beh2']=490
IdxFerm['h2o']=196
IdxFerm['nh3']=8008
listQubMets=['qwc','qwc_jw','fc','fc_jw','qwc_si','qwc_si_jw','fc_si','fc_si_jw']


logpath=PTRes+mol+'/'+meth+'/'

if not os.path.exists(logpath):
    os.makedirs(logpath)

log_file = open(logpath+'PTEstlogfile.txt', 'w')
sys.stdout = log_file


if states=='HF':

    h_ferm=sl.load_fermionic_hamiltonian(mol,prefix=path_prefix)


    fockOp=GetFockOp(h_ferm,eta)

    if meth in listQubMets:

        filename=pathTap+mol+'/'+meth+'/'+mol+'_'+meth+'_SymFrags.pk'
        f=open(filename,'rb')
        dat=pickle.load(f)
        CliffUnit=dat['CliffUnit']
        EigNums=dat['EigSyms']
        RedSymList=dat['RotSyms']

        encod=dat['encoding']


        if encod=='jw':
            qubOneBodHF=openfermion.jordan_wigner(fockOp)
        elif encod == 'bk':
            qubOneBodHF=openfermion.bravyi_kitaev(fockOp)
        else:
            print("Qubit encoding not recognized!!!")

        RotHam=openfermion.hermitian_conjugated(CliffUnit)*qubOneBodHF*CliffUnit

        TapOneBodHF=taps.TapperRotHam(EigNums,RedSymList,RotHam,nqubs)

        RemQubs=nqubs-len(RedSymList)
        ProjHam=openfermion.get_sparse_operator(TapOneBodHF,n_qubits=RemQubs)
    else:

        fock_sp=openfermion.get_sparse_operator(fockOp)


        SymUpath=SymFragPath+mol+'_v.npy'

        Vdag=sparse.csc_matrix(np.load(SymFragPath+mol+'_vdag.npy'))
        V=sparse.csc_matrix(np.load(SymFragPath+mol+'_v.npy'))

        ###projecting the fock operator to a symmetric subspace...
        fock_sp=Vdag*fock_sp*V

        ProjHam=fock_sp[0:IdxFerm[mol],0:IdxFerm[mol]]

else:

    if meth in listQubMets:
        FileName=pathTap+mol+'/'+meth+'/'+mol+'_'+meth+'_SymFrags.h5'

        f=h5py.File(FileName,'r')

        NFrags=int(len(f[meth].keys())/4)
        f.close()

        Frags=LoadAllFragsTap(FileName,meth,NFrags)

    else:

        Frags=LoadAllFrags(mol,meth,SymMatrixPath)

    ProjHam=Frags[0]

    for i in range(1,len(Frags)):

        ProjHam+=Frags[i]


#Diagonalization of projected Hamiltonian
Eigs,Eigvects=np.linalg.eigh(ProjHam.toarray())


###Loading the C1Op and C2Op files..

C1Name=PTRes+mol+'/'+meth+'/'+mol+'_'+meth+'C1OPGSSym.pk'
C2Name=PTRes+mol+'/'+meth+'/'+mol+'_'+meth+'C2OPGSSym.pk'

#Processing C2Op first
f=open(C2Name,'rb')
dat=pickle.load(f)

try:
    COp = dat['C2Op'].item()
except AttributeError:
    COp = dat['C2Op']

f.close()

GS=Eigvects[:,0]

EPT1=np.dot(np.conjugate(np.transpose(GS)),COp.dot(GS))/3.0

print("Perturbative correction to eigenvalue due to C2 Op for states", states, "is",EPT1)

f=open(C1Name,'rb')
dat=pickle.load(f)
COp=dat['C1Op']
f.close()


EPT2 = PT2Est(Eigs,Eigvects,COp)

print("Perturbative correction to eigenvalue due to C1 Op for states", states, "is",EPT2)


savefile=PTRes+mol+'/'+meth+'/'+mol+'_'+meth+'Pert'+states+'.pk'
Dict={}
Dict['EPT1']=EPT1
Dict['EPT2']=EPT2

with open(savefile, 'wb') as handle:
    pickle.dump(Dict, handle)


log_file.close()
