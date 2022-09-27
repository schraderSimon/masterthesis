import sys
sys.path.append("../libraries")
from rccsd_gs import *

from func_lib import *
from numba import jit
from matrix_operations import *
from helper_functions import *
basis = 'cc-pVTZ'
charge = 0
molecule=lambda x:  "H 0 0 0; F 0 0 %f"%x
refx=[1.75]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geometry=np.linspace(1.5,4.5,4)
import pickle
geom_alphas1=np.linspace(1.2,5,39)
geom_alphas=[[x] for x in geom_alphas1]
energy_dict={}
sample_geom1=sample_geometry
sample_geom=[[x] for x in sample_geom1]
sample_geom1=np.array(sample_geom).flatten()
t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis,reference_determinant,mix_states=False,type="procrustes")
evcsolver=EVCSolver(geom_alphas,molecule,basis,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False)
E_WF=evcsolver.solve_WFCCEVC()

E_AMP=evcsolver.solve_AMP_CCSD(occs=1,virts=1)
print("WF")
print(list(E_WF))
print("AMP")
print(list(E_AMP))
charge = 0
molecule=lambda x:  "H 0 0 0; F 0 0 %f"%x
refx=[1.75]
print(molecule(*refx))
reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
sample_geometry=[np.linspace(1.5,5,6)]
geom_alphas1=np.linspace(1.2,5,39)
CCSD=[-100.14885459358051, -100.23580451413115, -100.2901331075471, -100.32209866524957, -100.33867128519431, -100.34464174181993, -100.34333846786548, -100.33709973295649, -100.32758466402635, -100.31597807003708, -100.30312756844954, -100.28963814526753, -100.27593942701652, -100.26233467331497, -100.2490367649001, -100.23619438795419, -100.22391067723528, -100.21225596387359, -100.20127646754516, -100.19100007700061, -100.18144014288673, -100.1725982203032, -100.1644658919701, -100.15702641685095, -100.15025602946896, -100.14412525485567, -100.13860002850176, -100.13364290959709, -100.12921420689494, -100.12527302774745, -100.12177820607486, -100.11868898808851, -100.11596601394353, -100.1135716040041, -100.11147012081184, -100.10962853397163, -100.10801645655015, -100.1066061195395, -100.10537245472662]
x=np.linspace(1.2,5,39)
plt.plot(geom_alphas1,CCSD,label="CCSD")
plt.ylabel("Energy (Hartree)")
plt.xlabel("x (Bohr)")
plt.xticks(np.linspace(2,5,4))

plt.show()
