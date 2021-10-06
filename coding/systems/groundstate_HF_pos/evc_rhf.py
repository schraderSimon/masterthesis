import sys
sys.path.append("../../eigenvectorcontinuation/")

from REC import *
from matrix_operations import *
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
plt.figure(figsize=(9,6))
basis="6-31G"
sample_x=np.linspace(1.5,2.0,11)
xc_array=np.linspace(1.2,5.0,20)
molecule=lambda x: """F 0 0 0; H 0 0 %f"""%x
molecule_name=r"Hydrogen Fluoride"
'''
basis="6-31G*"
molecule_name="BeH2"
sample_x=np.linspace(0,2.5,151)
xc_array=np.linspace(3,4,11)
molecule=lambda x: """Be 0 0 0; H %f %f 0; H %f %f 0"""%(x,2.54-0.46*x,x,-(2.54-0.46*x))
'''
print("FCI")
#energiesFCI=FCI_energy_curve(xc_array,basis,molecule=molecule)
print("CCSDT")
energiesCC=CC_energy_curve(xc_array,basis,molecule=molecule)
start=timer()
for i in range(1,len(sample_x)+1,2):
    print("Eigvec (%d)"%(i))
    HF=eigvecsolver_RHF(sample_x[:i],basis,molecule=molecule,symmetry="C2v")
    energiesEC,eigenvectors=HF.calculate_energies(xc_array)
    plt.plot(xc_array,energiesEC,label="EC (%d point(s)), %s"%(i,basis))
end=timer()
print("Time elapsed: %f"%(end-start))
#In addition:"outliers":
#HF=eigvecsolver_RHF([2.5,3],basis,molecule=molecule)
#energiesEC,eigenvectors=HF.calculate_energies(xc_array)
#plt.plot(xc_array,energiesEC,label="EC (2.5 and 3)")
#print("UHF")

energiesHF=energy_curve_RHF(xc_array,basis,molecule=molecule)
energiesHF_sample=energy_curve_RHF(sample_x,basis,molecule=molecule)
sample_x=sample_x
xc_array=xc_array
plt.plot(xc_array,energiesHF,label="RHF,%s"%basis)
plt.plot(xc_array,energiesCC,label="CCSD(T),%s"%basis)
#plt.plot(xc_array,energiesFCI,label="FCI,%s"%basis)
#plt.plot(sample_x,energiesHF_sample,"o",color="black",label="Sample points")
plt.xlabel("Atomic distance (Bohr)")
plt.ylabel("Energy (Hartree)")
plt.title("%s potential energy curve"%molecule_name)
plt.legend(loc="lower right")
plt.tight_layout()
#plt.ylim([-100.3,-99.4])
plt.savefig("EC_RHF_%s.pdf"%molecule_name)
plt.show()
plt.plot(xc_array,energiesEC-energiesFCI,label="EC (max)-FCI")
plt.plot(xc_array,energiesHF-energiesFCI,label="RHF-FCI")
plt.legend()
plt.show()
