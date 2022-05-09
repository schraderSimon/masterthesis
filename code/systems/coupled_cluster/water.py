from rccsd_gs import *
import sys
sys.path.append("../libraries")

from func_lib import *
from matrix_operations import *
from helper_functions import *
np.set_printoptions(linewidth=300,precision=10,suppress=True)
from scipy.optimize import minimize, root,newton

def molecule(alpha,r1,r2):
    H2_x=r2*np.cos(np.pi*alpha/180)
    H2_y=r2*np.sin(np.pi*alpha/180)
    water="O 0 0 0; H %f 0 0; H %f %f 0"%(r1,H2_x,H2_y)
    return water

mix_states=False
basis="aug-cc-pVDZ"
molecule_name="Water"
ref_x=[104.5,1.81,1.81]
mol=make_mol(molecule,ref_x,basis)
ENUC=mol.energy_nuc()
mf=scf.RHF(mol)
mf.kernel()
rhf_mo_ref=mf.mo_coeff
n_samples=15 # How many samples to consider
number_repeats=5 #How often to repeat the experiment
WF_vals=[[] for i in range(0,number_repeats)]
AMP_vals=[[] for i in range(0,number_repeats)]
AMP_50=[[] for i in range(0,number_repeats)]
AMP_20=[[] for i in range(0,number_repeats)]
CCSD_vals=[]
for n in range(number_repeats):
    geom_alphas=[]
    sample_geom=[]
    i=0
    #Create n_samples sample geometries points
    while i<n_samples:
        angle=10 * np.random.random_sample() + 99.5
        r1=2.7*np.random.random_sample() + 1.3
        r2=2.7*np.random.random_sample() + 1.3

        mol=make_mol(molecule,[angle,r1,r2],basis)
        ENUC=mol.energy_nuc()
        mf=scf.RHF(mol)
        mf.kernel()
        if mf.converged:
            sample_geom.append([angle,r1,r2])
            i+=1
        else:
            pass
            print([angle,r1,r2])
    i=0
    #Create n_samples+10 test geometries
    while i<n_samples+1:
        angle=10 * np.random.random_sample() + 99.5
        r1=2.7*np.random.random_sample() + 1.3
        r2=2.7*np.random.random_sample() + 1.3

        mol=make_mol(molecule,[angle,r1,r2],basis)
        ENUC=mol.energy_nuc()
        mf=scf.RHF(mol)
        mf.kernel()
        if mf.converged:
            i+=1
            geom_alphas.append([angle,r1,r2])
        else:
            pass
            print([angle,r1,r2])
    t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis,rhf_mo_ref,mix_states)


    evcsolver=EVCSolver(geom_alphas,molecule,basis,rhf_mo_ref,t1s[:i],t2s[:i],l1s[:i],l2s[:i],sample_x=sample_geom[:i],mix_states=False)
    CCSD_energy=np.array(evcsolver.solve_CCSD())
    CCSD_vals.append(CCSD_energy)
    for i in range(1,n_samples+1,1):
        evcsolver=EVCSolver(geom_alphas,molecule,basis,rhf_mo_ref,t1s[:i],t2s[:i],l1s[:i],l2s[:i],sample_x=sample_geom[:i],mix_states=False)
        E_WF=np.array(evcsolver.solve_WFCCEVC())
        E_AMP_full=np.array(evcsolver.solve_AMP_CCSD(occs=1,virts=1))
        E_AMP_50=np.array(evcsolver.solve_AMP_CCSD(occs=1,virts=0.5))
        E_AMP_20=np.array(evcsolver.solve_AMP_CCSD(occs=1,virts=0.2))
        WF_vals[n].append(E_WF)
        AMP_vals[n].append(E_AMP_full)
        AMP_20[n].append(E_AMP_20)
        AMP_50[n].append(E_AMP_50)
data={}
data["CCSD"]=CCSD_vals
data["WF"]=WF_vals
data["AMP"]=AMP_vals
data["AMP20"]=AMP_20
data["AMP50"]=AMP_50

file="energy_data/water_631G*.bin"
import pickle
with open(file,"wb") as f:
    pickle.dump(data,f)


means_WF=[]
std_WF=[]
means_AMP=[]
std_AMP=[]
WF_quantiles=[]
AMP_quantiles=[]
from scipy.stats import mstats

for i in range(0,n_samples):
    WF_quantiles.append(mstats.mquantiles(errors_WF[i]))
    AMP_quantiles.append(mstats.mquantiles(errors_AMP[i]))
    means_WF.append(np.mean(errors_WF[i]))
    means_AMP.append(np.mean(errors_AMP[i]))
    std_WF.append(np.std(errors_WF[i]))
    std_AMP.append(np.std(errors_AMP[i]))
WF_quantiles=np.array(WF_quantiles)
AMP_quantiles=np.array(AMP_quantiles)

std_AMP=np.array(std_AMP)
means_WF=np.array(means_WF)
std_WF=np.array(std_WF)
means_AMP=np.array(means_AMP)
sample_points=np.arange(1,n_samples+1,dtype=int)
