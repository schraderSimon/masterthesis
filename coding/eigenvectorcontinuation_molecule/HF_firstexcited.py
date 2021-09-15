import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf, fci,cc,ao2mo, mp, mcscf,symm
import pyscf
import sys
from eigenvectorcontinuation import generalized_eigenvector
np.set_printoptions(linewidth=200,precision=4,suppress=True)

from ec_HF_basischange import *
ax,fig=plt.subplots(figsize=(10,10))

basis="STO-3G"
sample_x=np.linspace(1.5,1.6,2)
xc_array=np.linspace(1.5,1.6,2)
molecule=lambda x: """H 0 0 0; F 0 0 %f"""%x
molecule_name="HF"
energiesCC=CC_energy_curve(xc_array,basis,molecule=molecule)
for i in range(1,3):
    print("Eigvec (%d)"%(i))
    HF=eigensolver_RCI(sample_x[:i],basis,molecule=molecule,symmetry="coov")
    energiesEC,eigenvectors=HF.calculate_energies(xc_array)
    plt.plot(xc_array,energiesEC,label="EC (%d points), %s"%(i,basis))
#In addition:"outliers":
print("RHF")
energiesHF=energy_curve_RHF(xc_array,basis,molecule=molecule)
ymin=np.amin([energiesHF,energiesCC])
ymax=np.amax([energiesHF,energiesCC])
sample_x=sample_x
xc_array=xc_array
plt.plot(xc_array,energiesHF,label="RHF,%s"%basis)
plt.plot(xc_array,energiesCC,label="CCSD(T),%s"%basis)
#plt.plot(xc_array,energiesFCI,label="FCI,%s"%basis)
plt.vlines(sample_x,ymin,ymax,linestyle="--",color=["grey"]*len(sample_x),alpha=0.5,label="sample point")
plt.xlabel("Molecular distance (Bohr)")
plt.ylabel("Energy (Hartree)")
plt.title("%s potential energy curve"%molecule_name)
plt.legend()
plt.savefig("EC_RHF_%s.png"%molecule_name)
plt.show()
plt.plot(xc_array,energiesEC-energiesFCI,label="EC (max)-FCI")
plt.plot(xc_array,energiesHF-energiesFCI,label="RHF-FCI")
plt.legend()
plt.show()
