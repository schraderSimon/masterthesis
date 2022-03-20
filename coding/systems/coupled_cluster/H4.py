import matplotlib.pyplot as plt
from pyscf import gto, scf, cc
import numpy as np
from numpy import sin, cos,pi
from pyscf.symm.addons import symmetrize_orb
def molecule(theta):
	x=cos(theta)
	y=sin(theta)
	R=2

	#print("H %f 0 0; H -%f 0 0; H %f %f 0; H -%f -%f 0"%(R,R,R*x,R*y, R*y,R*x))
	return "H %f 0 0; H -%f 0 0; Li %f %f 0; F -%f -%f 0"%(R,R,R*x,R*y, R*x,R*y)

thetas=np.linspace(50,130,41)
E_HF=np.zeros(len(thetas))
E_CCSD=np.zeros(len(thetas))
MO_E=[]
for i,theta in enumerate(thetas):
	mol=gto.Mole(atom=molecule(pi/180*theta),basis="STO-3G")
	mol.build()
	mf=scf.RHF(mol)
	print(mol.intor("int1e_ovlp"))
	e_HF=mf.kernel()
	E_HF[i]=e_HF
	MO_E.append(mf.mo_energy)
plt.plot(thetas,E_HF,label="E_HF")
plt.show()
