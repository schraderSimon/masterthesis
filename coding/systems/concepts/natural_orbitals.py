import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
def molecule(x):
    return "F 0 0 0; H 0 0 %f"%x
def orthogonal_procrustes(mo_new,reference_mo):
    A=reference_mo
    B=mo_new.T
    M=B@A
    U,s,Vt=scipy.linalg.svd(M)
    return U@Vt, 0
xs=np.linspace(1.5,5,72)
x=1
basis="cc-pVTZ"
mol=gto.M(atom=molecule(x),basis=basis,spin=0,unit="bohr")
S=mol.intor("int1e_ovlp")
Sref=mol.intor("int1e_ovlp")
myhf=scf.RHF(mol)
e=myhf.run()
mymp=mp.RMP2(myhf).run()

rdm1 =mymp.make_rdm1()  # Add the correlation part
#noons,natorbs=scipy.linalg.eigh(rdm1)
#noons_ref=noons[::-1]
#natorbs_ref=linalg.fractional_matrix_power(Sref, -0.5)@(natorbs.T[::-1]).T
noons_ref,natorbs_ref=mcscf.addons.make_natural_orbitals(mymp)
noonss=[]
old_noonss=[]
natorbss=[]
'''
for i,col in enumerate((natorbs_ref[:,5:]).T):
    if (col[np.argmax(np.abs(col))]<0):
        natorbs_ref[:,5+i]=-col.copy()
'''
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
choices=np.random.choice(np.arange(len(noonss[0])),size=len(noonss[0])//1, replace=False)
circle2 = plt.Circle((5, 5), 0.5, color='b', fill=False)

axes[1].set_title("Not analytic")
axes[0].set_title("Analytic")
axes[0].plot(xs,np.array(noonss)[:,5:])
axes[1].plot(xs,np.array(old_noonss)[:,5:])
axes[0].set_xlabel("x (Bohr)")
axes[1].set_xlabel("x (Bohr)")
axes[0].set_ylabel("Natural occupation number")
#axes[1].set_ylabel("Natural occupation number")
axes[0].set_yscale("log")
axes[0].plot(3.35,0.0044,marker='o',markerfacecolor ='none',markeredgewidth=2,markeredgecolor="black",alpha=1.0,ms=x*5)
axes[0].plot(3.95,6*1e-4,marker='o',markerfacecolor ='none',markeredgewidth=2,markeredgecolor="black",alpha=1.0,ms=x*5)#,mec='None')

axes[0].set_ylim([3*1e-4,0.013])
axes[1].set_yscale("log")
plt.tight_layout()
plt.savefig("natorbs_dissosciation_HF.pdf")
plt.show()
