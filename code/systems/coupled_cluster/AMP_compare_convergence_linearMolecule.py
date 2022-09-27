"""
Get approximate AMP-CCEVC amplitudes for HF molecule in small basis
"""
import sys
sys.path.append("../libraries")
from rccsd_gs import *
from machinelearning import *
from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *
"""
def change_amplitude_basis(t1,t2,C_ref,C_new):
    #Step 1: Calculate U to go from Procrustes orbitals back to canonical orbitals.
    #This requires to make a small change to Øyvind's program such that it works with alternative schemes, e.g.
    # Step 2: "expand" t1 and t2 coefficients to 1-electron and 2-electron integrals
    # Step 3: Use ao2mo in pyscf to convert t1 and t2 coefficients to the "new" orbitals
"""
basis = 'cc-pVTZ'
#basis="6-31G*"
charge = 0
molecule=lambda x:  "H 0 0 0; F 0 0 %f"%x;molecule_name="HF"
#molecule=lambda x:  "Be 0 0 0; H 0 0 %f; H 0 0 -%f"%(x,x);molecule_name="BeH2"
refx=[1.75]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geometry=[[np.linspace(1.5,5,6),np.linspace(1.5,2,6)],[np.linspace(1.5,2,16),np.linspace(4.5,5,6)]]
sample_geom1=np.linspace(1.5,5.0,20)
import pickle
geom_alphas1=np.linspace(1.45,5.05,37)
geom_alphas=[[x] for x in geom_alphas1]
energy_dict={}
energy_dict["xval"]=geom_alphas1
energies_WF=[[],[]]
energies_AMP=[[],[]]
energies_AMPred=[[],[]]

sample_geom=[[x] for x in sample_geom1]
sample_geom1=np.array(sample_geom).flatten()
t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis,reference_determinant,mix_states=False,type="procrustes")
evcsolver=EVCSolver(geom_alphas,molecule,basis,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)

"""
Set up machine learning for t amplitudes
"""

t1s_orth,t2s_orth,coefs=orthonormalize_ts(evcsolver.t1s,evcsolver.t2s)
t1_machinelearn=[]
t2_machinelearn=[]
means=[]
for i in range(len(sample_geom1)):
    mean,std=multivariate_gaussian_gpy(sample_geom1,coefs[i],geom_alphas1,sigma=1,l=1)
    means.append(mean)
means=np.array(means)
for i in range(len(geom_alphas1)):
    t1_temp=np.zeros_like(t1s[0])
    t2_temp=np.zeros_like(t2s[0])
    for j in range(len(coefs)):
        t1_temp+=means[j][i]*t1s_orth[j]
        t2_temp+=means[j][i]*t2s_orth[j]
    t1_machinelearn.append(t1_temp)
    t2_machinelearn.append(t2_temp)
print("Initial")
xtol=1e-8 #Convergence tolerance
evcsolver.solve_CCSD_previousgeometry(xtol=xtol)
niter_prevGeom=evcsolver.num_iter

evcsolver.solve_CCSD_noProcrustes(xtol=xtol)
niter_CCSD=evcsolver.num_iter
evcsolver.solve_CCSD_startguess(t1_machinelearn,t2_machinelearn,xtol=xtol)
niter_machinelearn_guess=evcsolver.num_iter
E_AMP_red=evcsolver.solve_AMP_CCSD(occs=1,virts=0.3,xtol=1e-5)
t1s_reduced=evcsolver.t1s_final
t2s_reduced=evcsolver.t2s_final
evcsolver.solve_CCSD_startguess(t1s_reduced,t2s_reduced,xtol=xtol)
niter_AMP_startguess=evcsolver.num_iter

plt.vlines(sample_geom1,0,30,linestyle="dotted")
plt.plot(geom_alphas1,niter_CCSD,label="MP2")
plt.plot(geom_alphas1,niter_AMP_startguess,label="EVC")
plt.plot(geom_alphas1,niter_prevGeom,label="Previous geometry")
plt.plot(geom_alphas1,niter_machinelearn_guess,label="Gaussian Process")
plt.legend()
plt.tight_layout()
plt.xlabel("Distance (Bohr)")
plt.ylabel("Number of iterations")
plt.show()
# This "proves" that the machine learning approach works approximately as good as the sum approximation, as it gives us an "amazing" starting guess.
# The number of necessary CCSD calculations is then pretty low.
outdata={}
outdata["basis"]=basis
outdata["molecule_name"]=molecule_name
outdata["sample_geometries"]=sample_geom1
outdata["test_geometries"]=geom_alphas1
outdata["MP2"]=niter_CCSD
outdata["EVC"]=niter_AMP_startguess
outdata["prevGeom"]=niter_prevGeom
outdata["GP"]=niter_machinelearn_guess
file="energy_data/convergence_%s_%s_%d.bin"%(molecule_name,basis,len(sample_geom1))
import pickle
with open(file,"wb") as f:
    pickle.dump(energy_dict,f)
sys.exit(1)
