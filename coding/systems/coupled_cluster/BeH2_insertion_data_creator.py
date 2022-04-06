from rccsd_gs import *
import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
from matrix_operations import *
from helper_functions import *


basis = 'cc-pVDZ'
basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
molecule=lambda x: """Be 0 0 0; H %f %f 0; H %f %f 0"""%(x,2.54-0.46*x,x,-(2.54-0.46*x))

sample_coeffmatrices_1=[]
sample_coeffmatrices_2=[]
ground_coeffmatrices=[]
exc_coeffmatrices=[]
ground_energies=[]
sample_coeffmatrices_min=[]
totlen=81
xc_array=np.linspace(0,4,totlen)
occdict2={"A1":6,"B1":0,"B2":0}
occdict3={"A1":4,"B1":2,"B2":0}
occdict1={"A1":4,"B1":0,"B2":2}
occdicts=[occdict1,occdict2,occdict3]
energies=np.zeros((len(xc_array),3))
E_FCI=np.zeros(len(xc_array))
correct_ocdicts_sample1=[]
correct_ocdicts_sample2=[]
if basis=="cc-pVDZ":
    E_FCI_ref=[-15.83643, -15.836093, -15.834828, -15.832713, -15.82985, -15.826351, -15.822342, -15.817944, -15.813265, -15.808388, -15.80337, -15.798227, -15.792946, -15.787481, -15.781763, -15.77571, -15.769235, -15.762252, -15.75469, -15.746497, -15.737641, -15.728118, -15.717955, -15.707218, -15.696025, -15.684583, -15.673302, -15.663154, -15.656804, -15.658575, -15.667306, -15.67869, -15.690784, -15.702886, -15.714654, -15.72583, -15.736151, -15.745301, -15.752877, -15.758353, -15.761034]
    FCI_func=scipy.interpolate.interp1d(np.linspace(0,4,41),E_FCI_ref,kind="cubic")
    E_FCI=FCI_func(xc_array)
for k,x in enumerate(xc_array):
    print(x)
    atom=molecule(x)
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
    mo_coeff_temp=[]
    mo_en_temp=[]
    energies_temp=np.zeros(3)
    for i in [0,1,2]:
        mf = scf.RHF(mol)
        mf.verbose=0
        mf.irrep_nelec=occdicts[i]
        e=mf.kernel(verbose=0)
        mo_coeff_temp.append(mf.mo_coeff)
        mo_en_temp.append(mf.mo_energy)
        energies_temp[i]=e
    es=np.argsort(energies_temp)
    ground_energies.append(energies_temp[es[0]])
    if basis=="cc-pVDZ" or basis=="STO-3G":
        pass
    else:
        """
        mfci=fci.FCI(mf)
        e=mfci.kernel()[0]
        E_FCI[k]=e
        print("Pos: %f, EFCI: %f"%(x,e))
        """
        pass
    ground_coeffmatrices.append(mo_coeff_temp[es[0]])
    exc_coeffmatrices.append(mo_coeff_temp[es[1]])
    if x<2.5:
        for i in range(3):
            energies[k,i]=energies_temp[es[i]]
        sample_coeffmatrices_1.append(mo_coeff_temp[es[0]])
        sample_coeffmatrices_2.append(mo_coeff_temp[es[1]])
        correct_ocdicts_sample1.append(occdicts[es[0]])
        correct_ocdicts_sample2.append(occdicts[es[1]])
    else:
        for i in range(3):
            energies[k,i]=energies_temp[i]
        sample_coeffmatrices_1.append(mo_coeff_temp[0])
        sample_coeffmatrices_2.append(mo_coeff_temp[1])
        correct_ocdicts_sample1.append(occdicts[0])
        correct_ocdicts_sample2.append(occdicts[1])
print(correct_ocdicts_sample1)
print(correct_ocdicts_sample2)
data={}
data["FCI"]=E_FCI
sample_geom1=xc_array
sample_geom=[[x] for x in sample_geom1]
sample_geom1=np.array(sample_geom).flatten()
geom_alphas1=xc_array
geom_alphas=[[x] for x in geom_alphas1]

for i in range(len(sample_coeffmatrices_1)):
    new1=localize_procrustes(mol,sample_coeffmatrices_1[i],mf.mo_occ,ref_mo_coeff=sample_coeffmatrices_1[0],mix_states=False)
    sample_coeffmatrices_1[i]=new1
    new2=localize_procrustes(mol,sample_coeffmatrices_2[i],mf.mo_occ,ref_mo_coeff=sample_coeffmatrices_2[-1],mix_states=False)
    sample_coeffmatrices_2[i]=new2
data["sample_coeffmatrices_1"]=sample_coeffmatrices_1
data["sample_coeffmatrices_2"]=sample_coeffmatrices_2
approach_1_CC=[]

approach_2_CC=[]
t1s=[]
t2s=[]
l1s=[]
l2s=[]
energies=[]
t1s1=[]
t2s1=[]
l1s1=[]
l2s1=[]
energies1=[]
for i in range(len(xc_array)):
    atom=molecule(xc_array[i])
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
    mol.build()
    mf=scf.RHF(mol)
    mf.irrep_nelec=correct_ocdicts_sample1[i]
    e=mf.kernel(verbose=0)

    R=np.linalg.inv(mf.mo_coeff)@sample_coeffmatrices_1[i]
    R_occ=np.linalg.inv(R[:3,:3])
    R_unocc=np.linalg.inv(R[3:,3:])
    mycc = cc.CCSD(mf)
    mycc.conv_tol_normt = 1e-7
    mycc.level_shift=.2
    mycc.iterative_damping = 0.8
    if i==0:
        mycc.kernel()

    else:
        mycc.kernel(t1=t1,t2=t2)
    print(mycc.t1.shape)
    t1 = mycc.t1
    t2 = mycc.t2
    l1, l2 = mycc.solve_lambda()
    l1 = np.einsum("ia,ij,ab->jb",l1,R_occ,R_unocc)
    l2 = np.einsum("ijab,ik,jl,ac,bd->klcd",l2,R_occ,R_occ,R_unocc,R_unocc)
    t1s1.append(np.einsum("ia,ij,ab->jb",mycc.t1,R_occ,R_unocc))
    t2s1.append(np.einsum("ijab,ik,jl,ac,bd->klcd",mycc.t2,R_occ,R_occ,R_unocc,R_unocc))
    l1s1.append(l1)
    l2s1.append(l2)
    energies1.append(mycc.e_tot)
    print(mycc.e_tot)
data["CC_1"]=[t1s1,t2s1,l1s1,l2s1,energies1]

for i in range(len(xc_array)):
    atom=molecule(xc_array[::-1][i])
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
    mol.build()
    mf=scf.RHF(mol)
    mf.irrep_nelec=correct_ocdicts_sample2[::-1][i]
    e=mf.kernel(verbose=0)
    mf.mo_coeff=sample_coeffmatrices_2[::-1][i]
    mycc = cc.CCSD(mf)
    mycc.conv_tol_normt = 1e-7
    mycc.level_shift=.2
    mycc.iterative_damping = 0.8
    if i==0:
        mycc.kernel()

    else:
        mycc.kernel(t1=t1,t2=t2)
    t1 = mycc.t1
    t2 = mycc.t2
    l1, l2 = mycc.solve_lambda()
    l1 = np.einsum("ia,ij,ab->jb",l1,R_occ,R_unocc)
    l2 = np.einsum("ijab,ik,jl,ac,bd->klcd",l2,R_occ,R_occ,R_unocc,R_unocc)
    t1s.append(np.einsum("ia,ij,ab->jb",mycc.t1,R_occ,R_unocc))
    t2s.append(np.einsum("ijab,ik,jl,ac,bd->klcd",mycc.t2,R_occ,R_occ,R_unocc,R_unocc))
    t1s.append(t1)
    t2s.append(t2)
    l1s.append(l1)
    l2s.append(l2)
    energies.append(mycc.e_tot)
data["CC_2"]=[t1s,t2s,l1s,l2s,energies]
data["x"]=sample_geom
file="energy_data/BeH2_CCSD_rawdata.bin"
import pickle
with open(file,"wb") as f:
    pickle.dump(data,f)


sys.exit(1)
