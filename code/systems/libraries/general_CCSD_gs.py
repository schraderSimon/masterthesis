from qs_ref import *
import basis_set_exchange as bse
from coupled_cluster.ccsd import CCSD
from coupled_cluster.ccsd import rhs_t
from coupled_cluster.ccsd import energies as rhs_e
from opt_einsum import contract

from scipy.optimize import minimize, root,newton
import time
np.set_printoptions(linewidth=300,precision=8,suppress=True)

class sucess():
    """Mini helper class that resembles scipy's OptimizeResult."""
    def __init__(self,x,success,nfev):
        self.x=x
        self.success=success
        self.nfev=nfev
def orthonormalize_ts(t1s: list,t2s: list):
    """
    Given lists of t1 and t2 amplitudes, orthogonalizes the vectors t_1\osumt_2 using SVD in such a way that the span is the same.

    Input:
    t1s,t2s (lists): Lists of same lengths for t1 and t2 amplitudes

    returns:
    t1s,t2s (lists): Orthogonalized lists with the same span
    coefs (matrix): The linear combinations to express original elements in terms of new elements
    """
    t_tot=[]
    a,i=t1s[0].shape
    avg_norm=0
    for j in range(len(t1s)):
        #t1_contr=np.einsum("ai,bk->abik",t1s[j],t1s[j])
        #t_tot.append(np.concatenate((t1s[j],0.5*t1_contr+0.25*t2s[j]),axis=None))
        t_tot.append(np.concatenate((t1s[j],t2s[j]),axis=None))
        avg_norm+=np.sum(np.abs(t_tot[-1]))
    avg_norm/=len(t1s)
    t_tot=np.array(t_tot)
    t_tot_old=t_tot.copy()
    t_tot=t_tot.T
    U,s,Vt=svd(t_tot,full_matrices=False)
    #U,R=scipy.linalg.qr(t_tot,mode="economic")
    t_tot=(U@Vt).T
    t1_new=[]
    t2_new=[]
    coefs=np.zeros((len(t1s),len(t1s)))
    for j in range(len(t1s)):
        for k in range(len(t1s)):
            coefs[j,k]=t_tot_old[j,:]@t_tot[k,:]

    for j in range(len(t1s)):
        new_t1=np.reshape(t_tot[j,:a*i],(a,i))
        #new_t2=(4*np.reshape(t_tot[j,a*i:],(a,a,i,i))-2*np.einsum("ai,bk->abik",new_t1,new_t1))
        new_t2=np.reshape(t_tot[j,a*i:],(a,a,i,i))
        t1_new.append(new_t1)
        t2_new.append(new_t2)
    return t1_new,t2_new,coefs
def setUpsamples(sample_x,molecule_func,basis,rhf_mo_ref,mix_states=False,type="procrustes",weights=None):
    """
    Sets up lambda and t-amplitudes for a set of geometries.

    Input:
    sample_x (list): The geometries at which the molecule is constructed.
    molecule_func (function->String): A function which returns a string
        corresponding to a molecular geometry as function of parameters from sample_x.
    basis (string): The basis set.
    rhf_mo_ref (matrix or list of matrices):
        if matrix: Use Procrustes algorithm to use "best" HF reference, then calculate amplitudes based on that
        if list: Use coefficient matrices in the list directly to calculate amplitudes from.
    mix_states (bool): if general procrustes orbitals should be used.
    weights: Outdated parameter

    Returns:
    t1s,t2s,l1s,l2s: CCSD ampltidue and lambda equations
    sample_energies: CCSD Energies for the desired samples
    """
    t1s=[]
    t2s=[]
    l1s=[]
    l2s=[]
    sample_energies=[]
    for k,x in enumerate(sample_x):
        if isinstance(rhf_mo_ref,list):
            ref_state=rhf_mo_ref[k]
        else:
            ref_state=rhf_mo_ref
        system = construct_pyscf_system_rhf_ref(
            molecule=molecule_func(*x),
            basis=basis,
            add_spin=True,
            anti_symmetrize=True, #false?
            reference_state=ref_state,
        )
        ccsd = CCSD(system, verbose=False)
        ground_state_tolerance = 1e-12
        ccsd.compute_ground_state(
            t_kwargs=dict(tol=ground_state_tolerance),
            l_kwargs=dict(tol=ground_state_tolerance),
        )
        t, l = ccsd.get_amplitudes()
        t1s.append(t[0])
        t2s.append(t[1])
        l1s.append(l[0])
        l2s.append(l[1])
        sample_energies.append(ccsd.compute_energy().real+system.nuclear_repulsion_energy)
    return t1s,t2s,l1s,l2s,sample_energies
class EVCSolver():
    """
    Class to solve EVC equations. Contains both WF-CCEVC and AMP-CCEVC functions.
    Ini
    """
    def __init__(self,all_x,molecule_func,basis,reference_natorbs,t1s,t2s,l1s,l2s,givenC=False,sample_x=None,mix_states=False,natorb_truncation=None):
        self.all_x=all_x
        self.molecule_func=molecule_func
        self.sample_x=sample_x
        self.basis=basis
        self.reference_natorbs=reference_natorbs
        for i in range(len(t1s)):
            t1s[i]=np.array(t1s[i])
            t2s[i]=np.array(t2s[i])
            l1s[i]=np.array(l1s[i])
            l2s[i]=np.array(l2s[i])
        self.t1s=t1s
        self.t2s=t2s
        self.l1s=l1s
        self.l2s=l2s
        self.mix_states=mix_states
        self.natorb_truncation=natorb_truncation
        self.num_params=np.prod(t1s[0].shape)+np.prod(t2s[0].shape)
        self.coefs=None
        self.givenC=givenC
    def solve_CCSD(self):
        """
        Solves the CCSD equations.

        Returns:
        E_CCSD (list): CCSD Energies at all_x.
        """
        E_CCSD=[]
        for k,x_alpha in enumerate(self.all_x):
            if isinstance(self.reference_natorbs,list):
                ref_state=self.reference_natorbs[k]
            else:
                ref_state=self.reference_natorbs
            system = construct_pyscf_system_rhf_ref(
                molecule=self.molecule_func(*x_alpha),
                basis=self.basis,
                add_spin=True,
                anti_symmetrize=True,
                reference_state=ref_state,
                mix_states=self.mix_states,
                truncation=self.natorb_truncation
            )
            try:
                ccsd = CCSD(system, verbose=False)
                ground_state_tolerance = 1e-12
                ccsd.compute_ground_state(t_kwargs=dict(tol=ground_state_tolerance))
                E_CCSD.append(ccsd.compute_energy().real+system.nuclear_repulsion_energy)
            except:
                E_CCSD.append(np.nan)
        return E_CCSD


    def solve_WFCCEVC(self,filename=None,exponent=14):
        """
        Solves the AMP_CCSD equations.
        Input:
        filename (string): If filename is given, H & S matrices are stored to that file.
        exponent: The epsilon value exponent (epsilon=10^exponent) for solving the generalized Eigenvector problem
        Returns:
        E_WFCCEVC (list): WF-CCEVC Energies at all_x.
        """
        E_WFCCEVC=[]
        Hs=[]
        Ss= []
        for k,x_alpha in enumerate(self.all_x):
            if isinstance(self.reference_natorbs,list):
                ref_state=self.reference_natorbs[k]
            else:
                ref_state=self.reference_natorbs
            if self.givenC==False:
                system = construct_pyscf_system_rhf_ref(
                    molecule=self.molecule_func(*x_alpha),
                    basis=self.basis,
                    add_spin=True,
                    anti_symmetrize=True, #false?
                    reference_state=ref_state,
                    mix_states=self.mix_states,
                    truncation=self.natorb_truncation
                )
            else:
                system = construct_pyscf_system_rhf_ref(
                    molecule=self.molecule_func(*x_alpha),
                    basis=self.basis,
                    add_spin=False,
                    anti_symmetrize=True, #false?
                    givenC=ref_state,
                    mix_states=self.mix_states,
                    truncation=self.natorb_truncation
                )
            H,S=self._construct_H_S(system) #Get H and S as described in Ekstrøm & Hagen
            try:
                pass
                eigvals=np.real(scipy.linalg.eig(a=H,b=S+10**(-exponent)*scipy.linalg.expm(-S/10**(-exponent)))[0])
            except:
                eigvals=np.real(scipy.linalg.eig(a=H,b=S+np.eye(len(S))*10**(-exponent+1.5))[0])
            sorted=np.sort(eigvals)
            E_WFCCEVC.append(sorted[0])
            Hs.append(H)
            Ss.append(S)
        if filename is not None:
            dicty={}
            dicty["S"]=Ss
            dicty["H"]=Hs
            import pickle
            with open(filename,"wb") as f:
                pickle.dump(dicty,f)

        return E_WFCCEVC
    def _construct_H_S(self,system):
        """
        Constructs H and S matrix as described in Ekstrøm and Hagen.
        Input: quantum system containing molecular data and coefficient matrix

        Returns: H and S matrices as expressed in terms of the sample basis.
        """
        RHF_energy=system.compute_reference_energy().real
        H=np.zeros((len(self.t1s),len(self.t1s)))
        S=np.zeros((len(self.t1s),len(self.t1s)))
        for i in range(len(self.t1s)):
            f = system.construct_fock_matrix(system.h, system.u)
            t1_error = np.array(rhs_t.compute_t_1_amplitudes(f, system.u, self.t1s[i], self.t2s[i], system.o, system.v, np))
            t2_error = np.array(rhs_t.compute_t_2_amplitudes(f, system.u, self.t1s[i], self.t2s[i], system.o, system.v, np))
            exp_energy=rhs_e.compute_ccsd_ground_state_energy(f, system.u, self.t1s[i], self.t2s[i], system.o, system.v, np)+system.nuclear_repulsion_energy
            for j in range(len(self.t1s)):
                X1=self.t1s[i]-self.t1s[j]
                X2=self.t2s[i]-self.t2s[j]
                overlap=0
                overlap+=1+contract("ia,ai->",self.l1s[j],X1) #Nothing changes here as the l1 is multiplied by two from before
                overlap+=0.5*contract("ijab,ai,bj->",self.l2s[j],X1,X1)+0.25*contract("ijab,abij->",self.l2s[j],X2) # THE SUGGESTED SOLUTION
                S[i,j]=np.real(overlap)
                extra=contract("ia,ai->",self.l1s[j],t1_error)+contract("ijab,ai,bj->",self.l2s[j],X1,t1_error)+0.25*contract("ijab,abij->",self.l2s[j],t2_error) #Extra 0.5...?
                H[i,j]=np.real(overlap*exp_energy)
                H[i,j]+=np.real(extra)
                print(np.real(extra))
        print(S)
        print(H)
        sys.exit(1)
        return H,S
def get_reference_determinant(molecule_func,refx,basis,charge=0):
    mol = gto.Mole()
    mol.unit = "bohr"
    mol.charge = charge
    mol.cart = False
    mol.build(atom=molecule_func(*refx), basis=basis)
    hf = scf.RHF(mol)
    hf.kernel()
    return np.asarray(hf.mo_coeff)
if __name__=="__main__":
    basis = 'cc-pVDZ'
    charge = 0
    molecule=lambda x:  "Be 0 0 0; H 0 0 %f; H 0 0 -%f"%(x,x)
    refx=[2]
    print(molecule(*refx))
    reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
    sample_geom1=np.linspace(2,4,3)
    #sample_geom1=[2.5,3.0,6.0]
    sample_geom=[[x] for x in sample_geom1]
    sample_geom1=np.array(sample_geom).flatten()
    geom_alphas1=np.linspace(1,4,31)
    geom_alphas=[[x] for x in geom_alphas1]

    t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis,reference_determinant,mix_states=False,type="procrustes")
    evcsolver=EVCSolver(geom_alphas,molecule,basis,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False,natorb_truncation=None)
    E_WF=evcsolver.solve_WFCCEVC()
    E_CCSDx=evcsolver.solve_CCSD()
    plt.plot(geom_alphas1,E_CCSDx,label="CCSD")
    plt.plot(geom_alphas1,E_WF,label="WF-CCEVC")
    plt.tight_layout()
    plt.legend()
    plt.show()
