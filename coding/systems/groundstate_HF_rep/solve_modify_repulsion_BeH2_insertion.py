import sys
sys.path.append("../../eigenvectorcontinuation/")
import matplotlib
from REC import *
from matrix_operations import *
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import pickle
def molecule(x):
    y = lambda x: 2.54 - 0.46*x
    atom="H  " + str(-y(x)) + " 0 " + str(x) + "; H " + str(y(x)) + " 0  " + str(x) + "; Be 0 0 0"
    return atom
basis="6-31G*"
sample_strengths=np.linspace(1.1,0.5,13)
xc_array=np.linspace(0,4,41)
file="BeH2_data_MO_coefs_%s.bin"%basis

try:
    with open(file,"rb") as f:
        data=pickle.load(f)
    mo_coeffs_allgeometries=data["MOs"]
except:
    mo_coeffs_allgeometries=[]
    occdict2={"A1":6,"B1":0,"B2":0}
    occdict3={"A1":4,"B1":2,"B2":0}
    occdict1={"A1":4,"B1":0,"B2":2}
    occdicts=[occdict1,occdict2,occdict3]

    for k,x in enumerate(xc_array):
        atom=molecule(x)
        mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
        twentysix_MO_COEFF=[]
        for styrke in sample_strengths:
            mo_coeff_temp=[]
            mo_en_temp=[]
            energies_temp=np.zeros(3)
            for i in [0,1,2]:
                print(x,styrke,i)
                mf = scf.RHF(mol)
                mf.verbose=0
                mf.irrep_nelec=occdicts[i]
                eri=mol.intor('int2e',aosym="s1")*styrke
                mf._eri = ao2mo.restore(1,eri,mol.nao_nr())
                mol.incore_anyway=True
                mf.kernel()
                e=mf.kernel(verbose=0)
                mo_coeff_temp.append(mf.mo_coeff)
                energies_temp[i]=e
            es=np.argsort(energies_temp)
            twentysix_MO_COEFF.append(mo_coeff_temp[es[0]])
            twentysix_MO_COEFF.append(mo_coeff_temp[es[1]])
        mo_coeffs_allgeometries.append(twentysix_MO_COEFF)
    data={}
    data["x"]=xc_array
    data["MOs"]=mo_coeffs_allgeometries
    with open(file,"wb") as f:
        pickle.dump(data,f)
all_energies=[]
for i in range(13,1,-3):
    energies=[]
    for k,x in enumerate(xc_array):
        atom=molecule(x)
        mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
        mol.build()
        MOS_to_use=mo_coeffs_allgeometries[k][:2*i]
        MOS_to_use_new=[]
        for j in range(len(MOS_to_use)):
            MOS_to_use_new=MOS_to_use[j][:,:3] #Only occupied orbitals...
            print(MOS_to_use_new)
        eigvecsolver=eigensolver_RHF_knowncoefficients(MOS_to_use_new,basis,molecule=molecule)
        S,T=eigvecsolver.calculate_ST_matrices(mol,MOS_to_use_new)
        e,eigvec=generalized_eigenvector(T,S,threshold=1e-12)
        energies.append(e)
    all_energies.append(energies)
for energies in all_energies:
    plt.plot(xc_array,energies)
plt.show()
