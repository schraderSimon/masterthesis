import sys
sys.path.append("../../eigenvectorcontinuation/")
import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)
from REC import *
from matrix_operations import *
from helper_functions import *
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
fig, ax = plt.subplots(1,2,figsize=(8.27,(8.27/2)),sharex=True,sharey=True)
basis="6-31G*"
sample_x=[]
sample_x.append(np.linspace(2.75,3.0,5))
sample_x.append(np.linspace(1.5,2.0,5))
xc_array=np.linspace(1.2,4.5,50)
molecule=lambda x: """F 0 0 0; H 0 0 %f"""%x
molecule_name=r"Hydrogen Fluoride"
print("CCSDT")
energiesCC=CC_energy_curve(xc_array,basis,molecule=molecule)
energiesHF=energy_curve_RHF(xc_array,basis,molecule=molecule)

start=timer()
for pl_ind,sx in enumerate(sample_x):
    for i in range(1,len(sample_x[0])+1,2):
        print("Eigvec (%d)"%(i))
        HF=eigvecsolver_RHF(sx[:i],basis,molecule=molecule)
        energiesEC,eigenvectors=HF.calculate_energies(xc_array)
        if(pl_ind==0):
            ax[pl_ind].plot(xc_array,energiesEC,label="EC (%d point(s))"%(i))
        else:
            ax[pl_ind].plot(xc_array,energiesEC)
    energiesHF_sample=energy_curve_RHF(sx,basis,molecule=molecule)
    if pl_ind==0:
        ax[pl_ind].plot(sx,energiesHF_sample,"*",color="black",label="Sample points")
        ax[pl_ind].plot(xc_array,energiesHF,label="RHF")
        ax[pl_ind].plot(xc_array,energiesCC,label="CCSD(T)")
    else:
        ax[pl_ind].plot(sx,energiesHF_sample,"*",color="black")
        ax[pl_ind].plot(xc_array,energiesHF)
        ax[pl_ind].plot(xc_array,energiesCC)
end=timer()
print("Time elapsed: %f"%(end-start))
energiesHF=energy_curve_RHF(xc_array,basis,molecule=molecule)

sample_x=sample_x
xc_array=xc_array
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
#plt.xlabel("Internuclear distance (Bohr)")
#plt.ylabel("E (Hartree)")
fig.text(0.5, 0.04, 'Internuclear distance (Bohr)', ha='center')
fig.text(0.04, 0.5, 'E (Hartree)', va='center', rotation='vertical')

fig.suptitle("Potential energy curve: %s, basis: %s"%(molecule_name,basis))
fig.legend(handletextpad=0.1,
            labelspacing=0.1,
            handlelength=1,
            loc="center right",   # Position of legend
            borderaxespad=0.1,    # Small spacing around legend box
            title="Legend"  # Title for the legend
)
plt.tight_layout()

plt.subplots_adjust(right=0.78)
#plt.ylim([-100.3,-99.4])
plt.savefig("../../master_thesis_plots/EC_RHF_%s_%s.pdf"%(molecule_name,basis))
plt.show()
plt.plot(xc_array,energiesEC-energiesFCI,label="EC (max)-FCI")
plt.plot(xc_array,energiesHF-energiesFCI,label="RHF-FCI")
plt.legend()
plt.show()
