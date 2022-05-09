from rccsd_gs import *
import sys
sys.path.append("../libraries")
from func_lib import *
from matrix_operations import *
from helper_functions import *

"""
MATRICES ARE NOT SAVED AUTOMATICALLY. THIS REQUIRES A CHANGE TO RCCSD_GS!!!
"""

molecule=lambda x: """Be 0 0 0; H %f %f 0; H %f %f 0"""%(x,2.54-0.46*x,x,-(2.54-0.46*x))
basis = 'cc-pVDZ'
basis_set = bse.get_basis(basis, fmt='nwchem')

file="energy_data/BeH2_CCSD_rawdata.bin"
import pickle
with open(file,"rb") as f:
    data=pickle.load(f)
print("FCI")
print(list(data["FCI"]))
t1s_1,t2s_1,l1s_1,l2s_1,sample_energies_1=data["CC_1"]
print("Left")
print(list(sample_energies_1))
t1s_2,t2s_2,l1s_2,l2s_2,sample_energies_2=data["CC_2"]
t1s_2=t1s_2[::-1]; t2s_2=t2s_2[::-1]; l2s_2=l2s_2[::-1];l1s_2=l1s_2[::-1];sample_energies_2=sample_energies_2[::-1]
print("Right")

print(list(sample_energies_2))
for i in range(len(t1s_1)):
    t1s_1[i]=t1s_1[i].swapaxes(1,0)
    t2s_1[i]=t2s_1[i].swapaxes(2,1).swapaxes(3,0)
    t1s_2[i]=t1s_2[i].swapaxes(1,0)
    t2s_2[i]=t2s_2[i].swapaxes(2,0).swapaxes(3,1)
    l1s_1[i]*=2
    l1s_2[i]*=2#l2s_2[i]*2
    l2s_1[i]*=2#l2s_1[i]*2
    l2s_2[i]*=2#l2s_2[i]*2
    energies_FCI=data["FCI"]
guesses=[t1s_2,t2s_2,l1s_2,l2s_2]
charge = 0

sample_coeffmatrices_1=[]
sample_coeffmatrices_2=[]
ground_coeffmatrices=[]
exc_coeffmatrices=[]
ground_energies=[]
sample_coeffmatrices_min=[]
xc_array=np.linspace(2,4,10)
occdict2={"A1":6,"B1":0,"B2":0}
occdict3={"A1":4,"B1":2,"B2":0}
occdict1={"A1":4,"B1":0,"B2":2}
occdicts=[occdict1,occdict2,occdict3]
energies=np.zeros((len(xc_array),3))
E_FCI=np.zeros(len(xc_array))

sample_coeffmatrices_1=data["sample_coeffmatrices_1"]
sample_coeffmatrices_2=data["sample_coeffmatrices_2"]

sample_geom=data["x"]

t1s,t2s,l1s,l2s,sample_energies=setUpsamples_givenC(sample_geom,molecule,basis,givenCs=sample_coeffmatrices_2,guesses=guesses)
print("Sample energies")
print(list(sample_energies))
evcsolver=EVCSolver(sample_geom,molecule,basis,sample_coeffmatrices_2,t1s,t2s,l1s,l2s,givenC=True,sample_x=sample_geom)
E_WF=evcsolver.solve_WFCCEVC(filename="energy_data/newBeH2data_2.bin")

plt.plot(sample_geom,E_WF,label="CCSD WF left")
plt.plot(sample_geom,sample_energies,label="sample energies")

#plt.show()
"""
refx=[4]
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geom1=np.linspace(3,4,30)
#sample_geom1=[2.5,3.0,6.0]
sample_geom=[[x] for x in sample_geom1]
sample_geom1=np.array(sample_geom).flatten()

t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis_set,reference_determinant,mix_states=False,type="procrustes")
evcsolver=EVCSolver(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False,natorb_truncation=None)
E_WF=evcsolver.solve_WFCCEVC()


file="energy_data/BeH2_CCSD_rawdata.bin"
import pickle
with open(file,"rb") as f:
    data=pickle.load(f)
t1s_1,t2s_1,l1s_1,l2s_1,sample_energies_1=data["CC_1"]
for i in range(len(t1s_1)):
    t1s_1[i]=t1s_1[i].swapaxes(1,0)
    t2s_1[i]=t2s_1[i].swapaxes(2,1).swapaxes(3,0)
    l1s_1[i]*=2
    l1s_2[i]*=2#l2s_2[i]*2
    t1s_2[i]=t1s_2[i].swapaxes(1,0)
    t2s_2[i]=t2s_2[i].swapaxes(2,0).swapaxes(3,1)
    l2s_1[i]*=2#l2s_1[i]*2
    l2s_2[i]*=2#l2s_2[i]*2
    energies_FCI=data["FCI"]


sample_geom=data["x"]
sample_coeffmatrices_1=data["sample_coeffmatrices_1"]
#sample_coeffmatrices_2=data["sample_coeffmatrices_2"]

system = construct_pyscf_system_rhf_ref(
    molecule=molecule(0.8),
    basis=basis,
    add_spin=False,
    anti_symmetrize=False,
    givenC=sample_coeffmatrices_1[4],
)
rccsd = RCCSD(system, verbose=False)
ground_state_tolerance = 1e-9
rccsd.compute_ground_state(
    t_kwargs=dict(tol=ground_state_tolerance),
    l_kwargs=dict(tol=ground_state_tolerance),

)
t, l = rccsd.get_amplitudes()
t1_new=t[0]
t2_new=t[1]
l1_new=l[0]
l2_new=l[1]
print(np.linalg.norm(t1_new-t1s_1[4]))
print(np.linalg.norm(t2_new-t2s_1[4]))
print("break")
print(l1_new)
print(np.linalg.norm(l1_new-l1s_1[4]))
print(l2_new)
print(np.linalg.norm(l2_new))
sample_geom=sample_geom[::-1]
evcsolver=EVCSolver(sample_geom[:1],molecule,basis_set,sample_coeffmatrices_1[:1],t1s_1[:1],t2s_1[:1],l1s_1[:1],l2s_1[:1],givenC=True,sample_x=sample_geom[:1],mix_states=False,natorb_truncation=None)
E_WF=evcsolver.solve_WFCCEVC()
print(E_WF)
sys.exit(1)
plt.plot(sample_geom,E_WF,label="EVC")
plt.plot(sample_geom,sample_energies_1,label="1")
plt.plot(sample_geom,sample_energies_2,label="2")
plt.plot(sample_geom,energies_FCI,label="FCI")
plt.legend()
plt.show()
"""
