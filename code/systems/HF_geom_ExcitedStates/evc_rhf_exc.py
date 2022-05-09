import sys
sys.path.append("../libraries")
import matplotlib
from REC import *
from matrix_operations import *
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

basis="cc-pVDZ"
sample_geometry=[[np.linspace(1.5,2.0,3),np.linspace(1.5,5.0,3)],[np.linspace(1.5,2.0,3),np.linspace(1.5,5,3)]]
fig,axes=plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,10))
xc_array=np.linspace(1.2,5.0,39)
molecule=lambda x: """H 0 0 0; F 0 0 %f"""%x
molecule_name=r"Hydrogen Fluoride"
print("CCSD")
energiesCC=CC_energy_curve(xc_array,basis,molecule=molecule)
energiesHF=energy_curve_RHF(xc_array,basis,molecule=molecule)
types=["procrustes","transform"]
name=["Gen. Procrustes","Symm. Orth."]
for i in range(len(sample_geometry)):
    for j in range(len(sample_geometry)):
        sx=sample_geometry[i][j]
        HF=eigvecsolver_RHF_singles(sx,basis,molecule=molecule,type=types[i])
        energiesEC,eigenvectors=HF.calculate_energies(xc_array,write=True,filename="HF_%s_%d%d.bin"%(basis,i,j))
        energiesHF_sample=energy_curve_RHF(sx,basis,molecule=molecule)
print(energiesCC)
print(energiesHF)
