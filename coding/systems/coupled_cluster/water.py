from rccsd_gs import *
import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *
sys.path.append("../../eigenvectorcontinuation/")
np.set_printoptions(linewidth=300,precision=10,suppress=True)
from scipy.optimize import minimize, root,newton

def molecule(alpha,r1,r2):
    H2_x=r2*np.cos(np.pi*alpha/180)
    H2_y=r2*np.sin(np.pi*alpha/180)
    water="O 0 0 0; H %f 0 0; H %f %f 0"%(r1,H2_x,H2_y)
    return water

mix_states=False
basis="cc-pVDZ"
molecule_name="Water"
number_repeats=5
ref_x=[104.5,1.81,1.81]
mol=make_mol(molecule,ref_x,basis)
ENUC=mol.energy_nuc()
mf=scf.RHF(mol)
mf.kernel()
rhf_mo_ref=mf.mo_coeff
n_samples=15
errors_WF=[[] for i in range(0,n_samples)]
errors_AMP=[[] for i in range(0,n_samples)]
for n in range(number_repeats):
    geom_alphas=[]
    sample_geom=[]
    i=0
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
    while i<n_samples:
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
    i=1
    E_CCSD,E_approx,E_diffguess,E_RHF,E_ownmethod=solve_evc(geom_alphas,molecule,basis,rhf_mo_ref,t1s[:i],t2s[:i],l1s[:i],l2s[:i],mix_states=mix_states,run_cc=True,cc_approx=False)
    energy_simen=solve_evc2(geom_alphas,molecule,basis,rhf_mo_ref,t1s[:i],t2s[:i],l1s[:i],l2s[:i],mix_states=mix_states)
    E_CCSD=np.array(E_CCSD)
    E_WF=np.array(E_approx)
    E_AMP=np.array(energy_simen)
    error_AMP=list(np.abs(E_AMP-E_CCSD))
    error_WF=list(np.abs(E_WF-E_CCSD))
    errors_WF[i-1]=errors_WF[i-1]+error_WF
    errors_AMP[i-1]=errors_AMP[i-1]+error_AMP
    for i in range(2,n_samples+1,1):
        trash,E_approx,E_diffguess,E_RHF,E_ownmethod=solve_evc(geom_alphas,molecule,basis,rhf_mo_ref,t1s[:i],t2s[:i],l1s[:i],l2s[:i],mix_states=mix_states,run_cc=False,cc_approx=False)
        energy_simen=solve_evc2(geom_alphas,molecule,basis,rhf_mo_ref,t1s[:i],t2s[:i],l1s[:i],l2s[:i],mix_states=mix_states)
        E_CCSD=np.array(E_CCSD)
        E_WF=np.array(E_approx)
        E_AMP=np.array(energy_simen)
        error_AMP=list(np.abs(E_AMP-E_CCSD))
        error_WF=list(np.abs(E_WF-E_CCSD))
        errors_WF[i-1]=errors_WF[i-1]+error_WF
        errors_AMP[i-1]=errors_AMP[i-1]+error_AMP
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
plt.plot(sample_points,means_AMP,label="AMP error (mean)",color="b")
plt.plot(sample_points,means_WF,label="WF error (mean)",color="r")
plt.fill_between(sample_points,AMP_quantiles[:,0],AMP_quantiles[:,2],color='b', alpha=0.2,label=r"IQR (AMP)")
plt.fill_between(sample_points,WF_quantiles[:,0],WF_quantiles[:,2],color='r', alpha=0.2,label=r"IQR (WF)")
plt.legend()
plt.tight_layout()
plt.ylim([1e-5,0.1])
plt.yscale("log")
plt.ylabel("Enery deviation from CCSD ")
plt.xlabel("Number of sample points")
plt.tight_layout()
plt.savefig("resultsandplots/water_convergence.pdf")
plt.show()
print(means_WF)
print(std_WF)
print(errors_WF)
