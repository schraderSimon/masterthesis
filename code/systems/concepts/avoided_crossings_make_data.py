import sys
sys.path.append("../libraries")
from func_lib import *
def molecule(x):
    return "F 0 0 0; H 0 0 %f"%x
xs=np.linspace(1.2,5,80)
x=1.5
basis="cc-pVTZ"
mol=gto.M(atom=molecule(x),basis=basis,spin=0,unit="bohr")
S=mol.intor("int1e_ovlp")
Sref=mol.intor("int1e_ovlp")
myhf=scf.RHF(mol)



e=myhf.run()

orbital_energies_ref = np.diag(myhf.mo_coeff.T @ myhf.get_fock() @ myhf.mo_coeff)
canonical_orbs_ref=myhf.mo_coeff



mymp=mp.RMP2(myhf).run()

rdm1 =mymp.make_rdm1()  # Add the correlation part
noons_ref,natorbs_ref=mcscf.addons.make_natural_orbitals(mymp)
noonss=[]
old_noonss=[]
natorbss=[]
new_energies=[]
Rref=linalg.fractional_matrix_power(S, 0.5)@natorbs_ref #The underlying unitary obtained from S
new_energiess=[]
old_energiess=[]
fig,axes=plt.subplots(2,1,sharey=False,figsize=(7,10))
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
    orbital_energies = np.diag(myhf.mo_coeff.T @ myhf.get_fock() @ myhf.mo_coeff)
    old_energiess.append(orbital_energies)
    canonical_orbs=myhf.mo_coeff
    new_noons,new_natorbs=similiarize_natural_orbitals(noons_ref,natorbs_ref,noons,natorbs,mol.nelec,S,Sref)
    new_energies,new_canonOrbs=similiarize_canonical_orbitals(orbital_energies_ref,canonical_orbs_ref,orbital_energies,canonical_orbs,mol.nelec,S,Sref)
    noonss.append(new_noons)
    new_energiess.append(new_energies)
    print(new_energies[:len(new_energies)//2])
    old_noonss.append(noons)
    noons_ref=new_noons
    natorbs_ref=new_natorbs
    canonical_orbs_ref=new_canonOrbs
    orbital_energies_ref=new_energies
    natorbss.append(new_natorbs)
    Sref=S
choices=np.random.choice(np.arange(len(noonss[0])),size=len(noonss[0])//1, replace=False)
circle2 = plt.Circle((5, 5), 0.5, color='b', fill=False)

axes[0].set_title("Orbital energy (Hartree)")
axes[1].set_title("Natural occupation number")
axes[1].plot(xs,np.array(noonss)[:,5:])
axes[0].plot(xs,np.array(new_energiess)[:,5:])
axes[0].set_xlabel("x (Bohr)")
axes[1].set_xlabel("x (Bohr)")
axes[0].set_xticks([2,3,4,5])
axes[1].set_xticks([2,3,4,5])

noons_data=np.array(noonss)[:,5:]
energies_data=np.array(new_energiess)
avoided_crossing_locations=[[[1.67,1.5],[2.75,1]],[[3.35,0.0044],[3.95,6*1e-4]]]
crossing_locations=[[[1.95,2.45],[4.32,0.66]],[[3.095,0.000453],[4.32,0.000359]]]
energy_dict={}
energy_dict["noons"]=noons_data
energy_dict["energies"]=energies_data
energy_dict["crossings"]=crossing_locations
energy_dict["avoided_crossings"]=avoided_crossing_locations
energy_dict["xs"]=xs
import pickle
file="orbitals_data/avoided_crossings.bin"
with open(file,"wb") as f:
    pickle.dump(energy_dict,f)

axes[0].plot(1.67,1.5,marker='o',markerfacecolor ='none',markeredgewidth=2,markeredgecolor="black",alpha=1.0,ms=x*5) #Avoided crossing
axes[0].plot(2.75,1.0,marker='o',markerfacecolor ='none',markeredgewidth=2,markeredgecolor="black",alpha=1.0,ms=x*5) #Avoided rossing
axes[1].plot(3.35,0.0044,marker='o',markerfacecolor ='none',markeredgewidth=2,markeredgecolor="black",alpha=1.0,ms=x*5)
axes[1].plot(3.95,6*1e-4,marker='o',markerfacecolor ='none',markeredgewidth=2,markeredgecolor="black",alpha=1.0,ms=x*5)

axes[0].plot(1.95,2.45,marker='o',markerfacecolor ='none',markeredgewidth=2,markeredgecolor="fuchsia",alpha=1.0,ms=x*5) # crossing
axes[0].plot(4.32,0.66,marker='o',markerfacecolor ='none',markeredgewidth=2,markeredgecolor="fuchsia",alpha=1.0,ms=x*5) # crossing
axes[1].plot(3.095,0.000453,marker='o',markerfacecolor ='none',markeredgewidth=2,markeredgecolor="fuchsia",alpha=1.0,ms=x*5)
axes[1].plot(4.32,0.000359,marker='o',markerfacecolor ='none',markeredgewidth=2,markeredgecolor="fuchsia",alpha=1.0,ms=x*5)

#axes[1].set_ylim([3*1e-4,0.013])
axes[1].set_ylim([3*1e-4,0.013])

axes[0].set_ylim([-0.2,3])
axes[1].set_yscale("log")
plt.tight_layout()
plt.savefig("avoided_crossings.pdf")
plt.show()
