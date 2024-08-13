'''
The CSA fragments can be retrieved from the CSA-related folders.
We can generate QWC, FC and SVD fragments
and create a library with them by using this function script
Use: python Gen_FragLib mol1 mol2 ... moln
The CSA and GCSA fragments are assumed
to be computed and saved in the /csa_computed directory in the root
directory of the project
'''

#from openfermion import get_sparse_operator
#from multiprocessing import Pool
import os
import pickle
import openfermion
import numpy as np

#User-defined libraries...

path_prefix="../CSA-master/" #Top of the project, corresponding to CSA/
import sys
sys.path.append(path_prefix)
sys.path.append("./")

#From CSA library
import saveload_utils as sl
from qubit_utils import get_qwc_group, get_fc_group, get_greedy_grouping
import var_utils
import ferm_utils

#Own functions
#import Trotter_utils
import SVD_utils

def RenormLCUGFRO(FermFrag,NSO,const_shift=False):
    '''
    It is assumed that FermFrag is represented in terms of operators for spin orbitals!
    Input: FermFrag the fermionic two-body operator, NSO the number of spin orbitals
    Returns: 1) the renormalized two-body operator, and 2) the renormalization one-body operator
    '''

    Nsp=NSO//2

    RenOne=np.zeros([Nsp,Nsp])

    SptIdxs=np.arange(0,NSO,2)
    TwoTen=ferm_utils.get_two_body_tensor(FermFrag,NSO)

    idx_i=0
    for i in SptIdxs:
        idx_j=0
        for j in SptIdxs:
            for k in SptIdxs:
                RenOne[idx_i,idx_j]+=TwoTen[i,j,k,k]
            idx_j+=1
        idx_i+=1

    OneBodRen=ferm_utils.get_ferm_op(RenOne)

    if const_shift:
        shift=0.0
        for i in range(NSO):
            for j in range(NSO):
                shift+=TwoTen[i,i,j,j]
        return FermFrag-2*OneBodRen+0.25*shift, 2*OneBodRen
    else:

        return FermFrag-2*OneBodRen, 2*OneBodRen




def MakeFROLib(mol,path_prefix):
    Nmols=len(mol)
    geo=1.0
    method='CSA'

    if not os.path.isdir('./Frag_Lib/FRO'):
        os.mkdir('./Frag_Lib/FRO/')


    for i in range(Nmols):
        if mol[i]=='h2':
            alpha=2
        elif mol[i]=='lih':
            alpha=8
        elif mol[i]=='beh2':
            alpha=12
        elif mol[i]=='h2o':
            alpha=10
        elif mol[i]=='nh3':
            alpha=12
        else:
            print("Molecule FRO results not available!")
            exit()

        Results={}
        CSASols=sl.load_csa_sols(mol[i],geo,method,alpha,prefix=path_prefix)
        h_ferm=sl.load_fermionic_hamiltonian(mol[i],path_prefix)
        nqubits=openfermion.count_qubits(h_ferm)
        #groups,nqubits=SVD_utils.SVD_decomp(mol[i],path_prefix)
        Results['n_qubits']=nqubits
        Results['grouping']=CSASols['grouping']
        f=open('./Frag_Lib/FRO/'+mol[i]+'_FROFrags','wb')
        pickle.dump(Results,f)
        f.close()


def MakeSVDLib(mol,path_prefix):
    Nmols=len(mol)

    if not os.path.isdir('./Frag_Lib/svd'):
        os.mkdir('./Frag_Lib/svd/')


    for i in range(Nmols):
        Results={}
        groups=[]

        #groups,nqubits=SVD_utils.SVD_decomp(mol[i],path_prefix)
        h_ferm=sl.load_fermionic_hamiltonian(mol[i],path_prefix)
        nqubits=openfermion.count_qubits(h_ferm)
        Htbt=ferm_utils.get_chemist_tbt(h_ferm) #in spatial orbital basis
        one_body = var_utils.get_one_body_correction_from_tbt(h_ferm, Htbt)
        print("N. of qubits for OneBod:",openfermion.count_qubits(one_body))

        groups.append(one_body)
        SVDtbts=SVD_utils.SVDDecomp(Htbt)

        for j in range(len(SVDtbts)):
            groups.append(ferm_utils.get_ferm_op(SVDtbts[j]))


        #For the sake of verification...
        RecHam=openfermion.FermionOperator()
        for j in range(len(groups)):
            RecHam+=groups[j]

        print("Difference between reconstructed Hamiltonian and original",openfermion.normal_ordered(RecHam-h_ferm))

        Results['n_qubits']=nqubits
        Results['grouping']=groups
        f=open('./Frag_Lib/svd/'+mol[i]+'_svdFrags','wb')
        pickle.dump(Results,f)
        f.close()

def MakeQWCLib(mol,path_prefix):
    Nmols=len(mol)

    if not os.path.isdir('./Frag_Lib/qwc'):
        os.mkdir('./Frag_Lib/qwc/')


    for i in range(Nmols):
        Results={}
        HFerm=sl.load_fermionic_hamiltonian(mol[i],path_prefix)
        nqubits=openfermion.utils.count_qubits(HFerm)
        #openfermion.transforms.bravyi_kitaev
        h_bk = openfermion.transforms.bravyi_kitaev(HFerm)
        h_bk_noconstant = h_bk - h_bk.constant; h_bk_noconstant.compress()
        groups = get_qwc_group(h_bk_noconstant)

        Results['n_qubits']=nqubits
        Results['grouping']=groups
        f=open('./Frag_Lib/qwc/'+mol[i]+'_qwcFrags','wb')
        pickle.dump(Results,f)
        f.close()

def MakeFCLib(mol,path_prefix):
    Nmols=len(mol)

    if not os.path.isdir('./Frag_Lib/fc'):
        os.mkdir('./Frag_Lib/fc/')

    color_alg = 'lf'
    for i in range(Nmols):
        Results={}
        HFerm=sl.load_fermionic_hamiltonian(mol[i],path_prefix)
        nqubits=openfermion.utils.count_qubits(HFerm)
        #openfermion.transforms.bravyi_kitaev
        h_bk = openfermion.transforms.bravyi_kitaev(HFerm)
        h_bk_noconstant = h_bk - h_bk.constant; h_bk_noconstant.compress()
        groups = get_fc_group(h_bk_noconstant,color_alg=color_alg)

        Results['n_qubits']=nqubits
        Results['grouping']=groups
        f=open('./Frag_Lib/fc/'+mol[i]+'_fcFrags','wb')
        pickle.dump(Results,f)
        f.close()

def MakeFC_SI_Lib(mol,path_prefix):
    Nmols=len(mol)

    if not os.path.isdir('./Frag_Lib/fc_si'):
        os.mkdir('./Frag_Lib/fc_si/')

    #color_alg = 'lf'
    for i in range(Nmols):
        Results={}
        HFerm=sl.load_fermionic_hamiltonian(mol[i],path_prefix)
        nqubits=openfermion.utils.count_qubits(HFerm)
        #openfermion.transforms.bravyi_kitaev
        h_bk = openfermion.transforms.bravyi_kitaev(HFerm)
        h_bk_noconstant = h_bk - h_bk.constant; h_bk_noconstant.compress()
        #groups = get_fc_group(h_bk_noconstant,color_alg=color_alg)
        groups = get_greedy_grouping(h_bk_noconstant,commutativity='fc')

        Results['n_qubits']=nqubits
        Results['grouping']=groups
        f=open('./Frag_Lib/fc_si/'+mol[i]+'_fc_siFrags','wb')
        pickle.dump(Results,f)

        f.close()


def MakeQWC_SI_Lib(mol,path_prefix):
    Nmols=len(mol)

    if not os.path.isdir('./Frag_Lib/qwc_si'):
        os.mkdir('./Frag_Lib/qwc_si/')

    #color_alg = 'lf'
    for i in range(Nmols):
        Results={}
        HFerm=sl.load_fermionic_hamiltonian(mol[i],path_prefix)
        nqubits=openfermion.utils.count_qubits(HFerm)
        #openfermion.transforms.bravyi_kitaev
        h_bk = openfermion.transforms.bravyi_kitaev(HFerm)
        h_bk_noconstant = h_bk - h_bk.constant; h_bk_noconstant.compress()
        #groups = get_fc_group(h_bk_noconstant,color_alg=color_alg)
        groups = get_greedy_grouping(h_bk_noconstant,commutativity='qwc')

        Results['n_qubits']=nqubits
        Results['grouping']=groups
        f=open('./Frag_Lib/qwc_si/'+mol[i]+'_qwc_siFrags','wb')
        pickle.dump(Results,f)
        f.close()

def MakeLCUSVDLib(mol,path_prefix):
    Nmols=len(mol)

    if not os.path.isdir('./Frag_Lib/SVDLCU'):
        os.mkdir('./Frag_Lib/SVDLCU/')


    for i in range(Nmols):
        Results={}
        groups=[]
        h_ferm=sl.load_fermionic_hamiltonian(mol[i],path_prefix)
        nqubits=openfermion.count_qubits(h_ferm)

        Htbt=ferm_utils.get_chemist_tbt(h_ferm) #in spatial orbital basis
        one_body = var_utils.get_one_body_correction_from_tbt(h_ferm, Htbt)

        SVDtbts=SVD_utils.SVDDecomp(Htbt)
        NSVDFrags=len(SVDtbts)

        groups.append(one_body)
        #Phase of "unitarization"...
        for j in range(NSVDFrags):
            #shift of each fragment...
            Ref2Bod,shifOneBod=SVD_utils.TranstoRefOp(SVDtbts[j])
            groups.append(Ref2Bod)
            groups[0]+=shifOneBod



        #groups,nqubits=SVD_utils.SVD_decomp(mol[i],path_prefix)

        Results['n_qubits']=nqubits
        Results['grouping']=groups
        f=open('./Frag_Lib/SVDLCU/'+mol[i]+'_SVDLCUFrags','wb')
        pickle.dump(Results,f)
        f.close()
        #return groups

def MakeGFROLCULib(mol,path_prefix):
    Nmols=len(mol)
    geo=1.0
    method='PGCSA'

    if not os.path.isdir('./Frag_Lib/GFROLCU'):
        os.mkdir('./Frag_Lib/GFROLCU/')


    for i in range(Nmols):
        if mol[i]=='h2':
            alpha=2
        elif mol[i]=='lih':
            alpha=77
        elif mol[i]=='beh2':
            alpha=117
        elif mol[i]=='h2o':
            alpha=118
        elif mol[i]=='nh3':
            alpha=186
        else:
            print("Molecule GFRO results not available!")
            exit()

        Results={}
        GFROSols=sl.load_csa_sols(mol[i],geo,method,alpha,prefix=path_prefix)
        h_ferm=sl.load_fermionic_hamiltonian(mol[i],path_prefix)
        nqubits=openfermion.count_qubits(h_ferm)

        FermFrags=GFROSols['grouping']
        # Performing renormalization over all fragments and saving the resulting fragments..

        Nfrags=len(FermFrags)
        NewFrags=[]
        NewFrags.append(FermFrags[0]) #kinetic energy op.

        for j in range(1,Nfrags):
            Ren2Bod,RenOneBod=RenormLCUGFRO(FermFrags[j],nqubits)
            NewFrags.append(Ren2Bod)
            NewFrags[0]+=RenOneBod

        #Sanity check...
        RecHam=openfermion.FermionOperator()
        for j in range(len(NewFrags)):
            RecHam+=NewFrags[j]


        print("Difference between reconstructed Hamiltonian and exact:",openfermion.normal_ordered(RecHam-h_ferm))

        #Saving results...
        Results['n_qubits']=nqubits
        Results['grouping']=NewFrags
        f=open('./Frag_Lib/GFROLCU/'+mol[i]+'_GFROLCUFrags','wb')
        pickle.dump(Results,f)
        f.close()

if len(sys.argv) < 2:
    mol=['h2']
else:
    mol=[]
    for i in range(len(sys.argv)-1):
        mol.append(sys.argv[i+1])


#Building Fragment library. This will trigger an error if the directory already exists
#if not os.path.isdir('./Frag_Lib/'):
#    os.mkdir('./Frag_Lib/')



#MakeLCUSVDLib(mol,path_prefix) #Change this line accordingly to generate the library of the partition method needed.
#MakeFROLib(mol,path_prefix=path_prefix)
MakeGFROLCULib(mol,path_prefix)
#MakeLCUSVDLib(mol,path_prefix)
MakeSVDLib(mol,path_prefix)
