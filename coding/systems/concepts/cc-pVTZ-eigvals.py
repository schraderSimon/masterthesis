import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
def molecule(x):
    return "F 0 0 0; H 0 0 %f"%x


xs=np.linspace(1.2,4.5,67)
x=1
basis="aug-cc-pVTZ"
mol=gto.M(atom=molecule(x),basis=basis,spin=0,unit="bohr")
S=mol.intor("int1e_ovlp")
Sref=mol.intor("int1e_ovlp")
myhf=scf.RHF(mol)
e=myhf.run()
mymp=mp.RMP2(myhf).run()
noons_ref,natorbs_ref=mcscf.addons.make_natural_orbitals(mymp)
noonss=[]
old_noonss=[]
natorbss=[]

Rref=linalg.fractional_matrix_power(S, 0.5)@natorbs_ref

diff=np.zeros(len(xs))
diff2=np.zeros(len(xs))
fig,axes=plt.subplots(1,2,sharey=True)
axes[0].set_ylabel("Natural occupation number")
for i,x in enumerate(xs):
    mol=gto.M(atom=molecule(x),basis=basis,spin=0,unit="bohr")
    S=mol.intor("int1e_ovlp")
    myhf=scf.RHF(mol)
    myhf.conv_tol=1e-10
    e=myhf.run()
    mymp=mp.RMP2(myhf)
    mymp.conv_tol=1e-10
    mymp.run()
    noons,natorbs=mcscf.addons.make_natural_orbitals(mymp)
    R=linalg.fractional_matrix_power(S, 0.5)@natorbs
    k=6
    diff[i]=np.linalg.norm(R[:,k:k+1]+Rref[:,k:k+1])
    diff2[i]=np.linalg.norm(R[:,k:k+1]-Rref[:,k:k+1])
    new_noons,new_natorbs=similiarize_natural_orbitals(noons_ref,natorbs_ref,noons,natorbs,mol.nelec,S,Sref)
    noonss.append(new_noons)
    old_noonss.append(noons)
    noons_ref=new_noons
    natorbs_ref=new_natorbs
    natorbss.append(new_natorbs)
    Sref=S
choices=np.random.choice(np.arange(len(noonss[0])),size=len(noonss[0])//5, replace=False)
axes[0].set_title("HF")
axes[0].plot(xs,np.array(noonss)[:,choices])
axes[0].set_xlabel("Interatomic distance (Bohr)")
axes[0].set_ylabel("Natural occupation number")
axes[0].set_yscale("log")

def molecule(x):
    return "H 0 0 -%f; Be 0 0 0; H 0 0 %f"%(x,x)
mol=gto.M(atom=molecule(x),basis=basis,spin=0,unit="bohr")
S=mol.intor("int1e_ovlp")
Sref=mol.intor("int1e_ovlp")
myhf=scf.RHF(mol)
e=myhf.run()
mymp=mp.RMP2(myhf).run()
noons_ref,natorbs_ref=mcscf.addons.make_natural_orbitals(mymp)
noonss=[]
old_noonss=[]
natorbss=[]

Rref=linalg.fractional_matrix_power(S, 0.5)@natorbs_ref

diff=np.zeros(len(xs))
diff2=np.zeros(len(xs))
for i,x in enumerate(xs):
    mol=gto.M(atom=molecule(x),basis=basis,spin=0,unit="bohr")
    S=mol.intor("int1e_ovlp")
    myhf=scf.RHF(mol)
    myhf.conv_tol=1e-10
    e=myhf.run()
    mymp=mp.RMP2(myhf)
    mymp.conv_tol=1e-10
    mymp.run()
    noons,natorbs=mcscf.addons.make_natural_orbitals(mymp)
    R=linalg.fractional_matrix_power(S, 0.5)@natorbs
    k=6
    diff[i]=np.linalg.norm(R[:,k:k+1]+Rref[:,k:k+1])
    diff2[i]=np.linalg.norm(R[:,k:k+1]-Rref[:,k:k+1])
    new_noons,new_natorbs=similiarize_natural_orbitals(noons_ref,natorbs_ref,noons,natorbs,mol.nelec,S,Sref)
    noonss.append(new_noons)
    old_noonss.append(noons)
    noons_ref=new_noons
    natorbs_ref=new_natorbs
    natorbss.append(new_natorbs)
    Sref=S
choices=np.random.choice(np.arange(len(noonss[0])),size=len(noonss[0])//5, replace=False)
axes[1].set_yscale("log")
axes[1].set_title(r"Be$H_2$")
axes[1].set_xlabel("Interatomic distance (Bohr)")
axes[1].plot(xs,np.array(noonss)[:,choices])
plt.tight_layout()
plt.savefig("natorbs_dissosciation.pdf")
plt.show()
