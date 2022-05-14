"""
Create S and H matrix for BeH2. File needs to be changed manually to choose which reference data is used
"""
import sys
sys.path.append("../libraries")
from rccsd_gs import *
from func_lib import *
from matrix_operations import *
from helper_functions import *


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
guesses=[t1s_2,t2s_2,l1s_2,l2s_2] #start guesses based on pyscf (alone, it wouldn't converge...)
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
