import sys
sys.path.append("../../eigenvectorcontinuation/")
from matrix_operations import *
from helper_functions import *
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *


def localize_cholesky(mol,mo_coeffc,mo_occ):
    mo=cholesky_coefficientmatrix(mo_coeffc[:,mo_occ>0])
    mo=swappistan(mo)
    mo_coeffc[:,mo_occ>0]=np.array(mo)
    mo_unocc=cholesky_coefficientmatrix(mo_coeffc[:,mo_occ<=0])
    mo_unocc=swappistan(mo_unocc)
    mo_coeffc[:,mo_occ<=0]=np.array(mo_unocc)
    return mo_coeffc

molecule=lambda x: "F 0 0 0; H 0 0 %f"%x
#molecule=lambda x: """Be 0 0 0; H %f %f 0; H %f %f 0"""%(x,2.54-0.46*x,x,-(2.54-0.46*x))
basis="cc-pVTZ"
molecule_name="HF"
x_sol=np.linspace(1.3,4,28)
#x_sol=np.array([1.9,2.5])
ref_x=[2]
mol=make_mol(molecule,ref_x[0],basis)
neh=mol.nelectron//2
mfref=scf.RHF(mol)
mfref.kernel()
ref_coefficientmatrix=mfref.mo_coeff
ref_coefficientmatrix=localize_cholesky(mol,mfref.mo_coeff,mfref.mo_occ).copy()


#mfref.mo_coeff=ref_coefficientmatrix
mycc=cc.CCSD(mfref); mycc.kernel(); ref_t2=mycc.t2

norms_coefficientmatrix=np.zeros((5,len(x_sol)))
norms_T2=np.zeros((5,len(x_sol)))
overlap_to_HF=np.zeros((5,len(x_sol)))
E_CC=np.zeros((5,len(x_sol)))
E_CCcanon=np.zeros(len(x_sol))
for i,x in enumerate(x_sol):
    mol=make_mol(molecule,x,basis)
    mfnew=scf.RHF(mol)
    mfnew.kernel()
    mycc=cc.CCSD(mfnew); E_CCcanon[i]=mycc.kernel()[0];E_CCcanon[i]=mycc.e_tot;
    cholesky_coeff=localize_cholesky(mol,mfnew.mo_coeff,mfnew.mo_occ).copy()
    procrustes_coeff=localize_procrustes(mol,cholesky_coeff.copy(),mfnew.mo_occ,ref_coefficientmatrix,mix_states=False).copy()
    gen_procrustes_coeff=localize_procrustes(mol,cholesky_coeff.copy(),mfnew.mo_occ,ref_coefficientmatrix,mix_states=True).copy()
    transformed_coeff=basischange(ref_coefficientmatrix.copy(),mol.intor("int1e_ovlp"),neh).copy()
    gen_transformed_coeff=basischange(ref_coefficientmatrix.copy(),mol.intor("int1e_ovlp"),mol.nao_nr()).copy()
    mf_cholesky=scf.RHF(mol); mf_cholesky.kernel(); mf_cholesky.mo_coeff=cholesky_coeff
    mf_transformed=scf.RHF(mol); mf_transformed.kernel();  mf_transformed.mo_coeff=transformed_coeff
    mf_procrustes=scf.RHF(mol); mf_procrustes.kernel();  mf_procrustes.mo_coeff=procrustes_coeff
    mf_gen_procrustes=scf.RHF(mol); mf_gen_procrustes.kernel();  mf_gen_procrustes.mo_coeff=gen_procrustes_coeff
    mf_gen_transformed=scf.RHF(mol); mf_gen_transformed.kernel();  mf_gen_transformed.mo_coeff=gen_transformed_coeff
    mycc=cc.CCSD(mf_procrustes); E_CC[0,i]=mycc.kernel()[0];E_CC[0,i]=mycc.e_tot; procrustes_t2=mycc.t2
    mycc=cc.CCSD(mf_transformed); E_CC[1,i]=mycc.kernel()[0];E_CC[1,i]=mycc.e_tot; transformed_t2=mycc.t2
    mycc=cc.CCSD(mf_cholesky); E_CC[2,i]=mycc.kernel()[0];E_CC[2,i]=mycc.e_tot; cholesky_t2=mycc.t2
    mycc=cc.CCSD(mf_gen_procrustes); E_CC[3,i]=mycc.kernel()[0];E_CC[3,i]=mycc.e_tot; gen_procrucstes_t2=mycc.t2
    mycc=cc.CCSD(mf_gen_transformed); E_CC[4,i]=mycc.kernel()[0];E_CC[4,i]=mycc.e_tot; gen_transformed_t2=mycc.t2
    norms_coefficientmatrix[0,i]=norm(procrustes_coeff-ref_coefficientmatrix)
    norms_coefficientmatrix[1,i]=norm(transformed_coeff-ref_coefficientmatrix)
    norms_coefficientmatrix[2,i]=norm(cholesky_coeff-ref_coefficientmatrix)
    norms_coefficientmatrix[3,i]=norm(gen_procrustes_coeff-ref_coefficientmatrix)
    norms_coefficientmatrix[4,i]=norm(gen_transformed_coeff-ref_coefficientmatrix)
    overlap_to_HF[0,i]=np.linalg.det(get_Smat(mol.intor("int1e_ovlp"),mfnew.mo_coeff[:,:neh],procrustes_coeff[:,:neh]))**2
    overlap_to_HF[1,i]=np.linalg.det(get_Smat(mol.intor("int1e_ovlp"),mfnew.mo_coeff[:,:neh],transformed_coeff[:,:neh]))**2
    overlap_to_HF[2,i]=np.linalg.det(get_Smat(mol.intor("int1e_ovlp"),mfnew.mo_coeff[:,:neh],cholesky_coeff[:,:neh]))**2
    overlap_to_HF[3,i]=np.linalg.det(get_Smat(mol.intor("int1e_ovlp"),mfnew.mo_coeff[:,:neh],gen_procrustes_coeff[:,:neh]))**2
    overlap_to_HF[4,i]=np.linalg.det(get_Smat(mol.intor("int1e_ovlp"),mfnew.mo_coeff[:,:neh],gen_transformed_coeff[:,:neh]))**2
    norms_T2[0,i]=norm(procrustes_t2-ref_t2)
    norms_T2[1,i]=norm(transformed_t2-ref_t2)
    norms_T2[2,i]=norm(cholesky_t2-ref_t2)
    norms_T2[3,i]=norm(gen_procrucstes_t2-ref_t2)
    norms_T2[4,i]=norm(gen_transformed_t2-ref_t2)
    #print("Procrustes: %.10f"%norms_coefficientmatrix[0,i])
    #print("Transformed: %.10f"%norms_coefficientmatrix[1,i])
    #print("Cholesky: %.10f"%norms_coefficientmatrix[2,i])
    #print("Gen Procrustes: %.10f"%norms_coefficientmatrix[3,i])
    #print(" GenTransformed: %.10f"%norms_coefficientmatrix[4,i])
    print(overlap_to_HF[:,i])
labels=["Procrustes", "Sym. Ort.", "Choleksy", "G. Procrustes", "G. Sym. Ort."]
fig,axes=plt.subplots(2,2,sharey=False,sharex=True,figsize=(12,10))
axes[0,0].set_title(r"$||C(x)-C(x_{{ref}})||$")
axes[0,1].set_title(r"$||T_2(x)-T_2(x_{{ref}})||$")
axes[1,0].set_title(r"$\langle \Phi^{{SD}}|\Phi^{{HF}}\rangle$")
axes[1,1].set_title(r"$E_{CCSD}-E_{CCSD}^{canon.} (Hartree)$")
for i in range(len(labels)):
    axes[0,0].plot(x_sol,norms_coefficientmatrix[i,:],label=labels[i])
    axes[0,1].plot(x_sol,norms_T2[i,:],label=labels[i])
    axes[1,0].plot(x_sol,overlap_to_HF[i,:],label=labels[i])
    axes[1,1].plot(x_sol,E_CC[i,:]-E_CCcanon,label=labels[i])
handles, labels = axes[0][0].get_legend_handles_labels()
axes[1][0].set_xlabel("distance (Bohr)")
axes[1][1].set_xlabel("distance (Bohr)")
fig.legend(handles, labels,loc="upper right")
fig.tight_layout()
fig.subplots_adjust(right=0.82)
plt.savefig("HF_coefficient_norms.pdf")
plt.show()
