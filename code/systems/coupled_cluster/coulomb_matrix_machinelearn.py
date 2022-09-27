import sys
sys.path.append("../libraries")
from rccsd_gs import *
from func_lib import *
from numba import jit
from machinelearning import *
from matrix_operations import *
from helper_functions import *
periodic_table={"H":1,"Be":4,"Li":3,"C":6,"N":7,"O":8,"F":9, "Cl":17,"Na":11}
def coulomb_matrix_from_molecule(mol,x):
    molstring=mol(x);
    print(molstring)
    atoms=molstring.split(";")
    coulomb_matrix=np.zeros((len(atoms),len(atoms)))
    pos=np.zeros((len(atoms),3))
    charges=np.zeros(len(atoms))
    for i,atom in enumerate(atoms):
        name,a,b,c=atom.strip(" ").split(" ")
        charge=periodic_table[name]
        charges[i]=charge
        coulomb_matrix[i,i]=0.5*charge**2.4
        pos[i,0]=float(a); pos[i,1]=float(b); pos[i,2]=float(c)
    for i in range(len(atoms)):
        for j in range(i+1,len(atoms)):
            pos1=pos[i,:]
            pos2=pos[j,:]
            diff=pos1-pos2
            R12=np.sqrt(np.dot(diff,diff))
            coulomb_matrix[i,j]=coulomb_matrix[j,i]=charges[i]*charges[j]/R12
    return coulomb_matrix
def molecule(x):
    return "F 0 0 0; H 0 0 %f"%(x[0])
    return "Be 0 0 0; H 0 0 -%f; H 0 0 %f"%(x[0],x[1])
def coulomb_norm(molecule, x1,x2):
    C1=coulomb_matrix_from_molecule(molecule,x1)
    C2=coulomb_matrix_from_molecule(molecule,x2)
    frobenius_norm=np.linalg.norm(C1-C2)

    eigvals1,eigvec=np.linalg.eigh(C1)
    print(x2)
    print(C2)
    eigvals2,eigvec=np.linalg.eigh(C2)
    eigen_norm=np.linalg.norm(eigvals1-eigvals2)
    return frobenius_norm,eigen_norm,np.linalg.norm(x1[0]-x2[0])
xvals=np.linspace(2,10,50)
yvals=np.linspace(2,10,50)
frob=[]
eigs=[]
absvals=[]
for i in range(len(xvals)):
    frobby,eigy,absy=coulomb_norm(molecule,np.array([xvals[i],yvals[i]]),np.array([5,8]))
    frob.append(frobby)
    eigs.append(eigy)
    absvals.append(absy)
frob=np.array(frob)/frob[0]
eigs=np.array(eigs)/eigs[0]
absvals=np.array(absvals)/absvals[0]
plt.plot(xvals,frob,label="Frobenius Norm")
plt.plot(xvals,eigs,label="Eigen Norm")
plt.plot(xvals,absvals,label="Absolute value")
plt.legend()
plt.show()
