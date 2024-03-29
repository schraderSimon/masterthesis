from qs_ref import *
from quantum_systems.custom_system import construct_pyscf_system_rhf
import basis_set_exchange as bse
from coupled_cluster.rccsd import RCCSD, TDRCCSD
import coupled_cluster
import rhs_t
from coupled_cluster.rccsd import energies as rhs_e
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
def basischange_clusterOperator(U,t1,t2):
    "Use U as change of basis operator to go from basis t1 to a new t1-tilde, same for t2"
    n_virt=t1.shape[0]
    n_occ=t1.shape[1]
    U_occ=U[:n_occ,:n_occ] #Occupied orbital rotation
    U_virt=U[n_occ:,n_occ:] #Virtual orbital rotation
    new_t1 = contract('ij,ai,ab->bj', U_occ, t1, U_virt)
    new_t2 = contract("ik,jl,abij,ac,bd->cdkl",U_occ,U_occ,t2,U_virt,U_virt)
    return new_t1,new_t2
    return t1,t2
    #eri_ao = mol.intor('int2e')
    #eri_mo = ao2mo.incore.full(eri_ao, mo_coeff)

def orthonormalize_ts(t1s: list,t2s: list):
    """
    Given lists of t1 and t2 amplitudes, orthogonalizes the vectors t_1\osumt_2 using SVD
    Input:
    t1s,t2s (lists): Lists of same lengths for t1 and t2 amplitudes

    returns:
    t1s,t2s (lists): Orthogonalized lists with the same span
    coefs (matrix): The linear combinations to express original elements in terms of new elements
    """
    t_tot=[]
    a,i=t1s[0].shape
    for j in range(len(t1s)):
        t_tot.append(np.concatenate((t1s[j],t2s[j]),axis=None)) #   T1 \osum T2
    t_tot=np.array(t_tot)
    t_tot_old=t_tot.copy()
    t_tot=t_tot.T
    U,s,Vt=svd(t_tot,full_matrices=False)
    t_tot=(U@Vt).T #This is the unitary matrix closest to t_tot
    t1_new=[]
    t2_new=[]
    coefs=np.zeros((len(t1s),len(t1s))) #Coefficients to transform between the old and the new representations
    for j in range(len(t1s)):
        for k in range(len(t1s)):
            coefs[j,k]=t_tot_old[j,:]@t_tot[k,:]

    for j in range(len(t1s)):
        new_t1=np.reshape(t_tot[j,:a*i],(a,i))
        new_t2=np.reshape(t_tot[j,a*i:],(a,a,i,i))
        t1_new.append(new_t1)
        t2_new.append(new_t2)
    return t1_new,t2_new,coefs
def setUpsamples_canonicalOrbitals(all_x,molecule_func,basis,desired_samples=None):
    """
    Sets up canonical orbitals and the corresponding lambda and t-amplitudes for a set of geometries.
    Canonical orbitals need to be calculated for each sample_x, even though we only want canonical orbitals and cluster
    amplitudes for "desired_samples", because we need a sufficiently small distance between natural orbitals in order
    to categorize them correctly.

    Input:
    all_x (list): The geometries at which the molecule is constructed. Possibly the whole PES of interest
    molecule_func (function->String): A function which returns a string
        corresponding to a molecular geometry as function of parameters from all_x.
    basis (string): The basis set.
    desired_samples (list): List of integers corresponding to indices in all_x. CCSD equations are only solved for those.

    Returns:
    t1s,t2s,l1s,l2s: CCSD ampltidue and lambda equations
    sample_energies: CCSD Energies for the desired samples
    reference_natorb_list: Coefficient matrices for canonical orbitals at desired_samples
    reference_overlap_list: The overlap matrices S at desired_samples
    reference_noons_list: Orbital energies at desired_samples
    """

    if desired_samples is None:
        desired_samples=np.array(np.arange(0,len(all_x)),dtype=int)
    t1s=[]
    t2s=[]
    l1s=[]
    l2s=[]
    sample_energies=[]
    reference_natorb_list=[]
    reference_overlap_list=[]
    reference_noons_list=[]
    systems=[]
    natorbs_ref=noons_ref=S_ref=None
    for k,x in enumerate(all_x[:]):
        system, natorbs_ref,noons_ref,S_ref = construct_pyscf_system_rhf_canonicalorb(
            molecule=molecule_func(*x),
            basis=basis,
            add_spin=False,
            anti_symmetrize=False,
            reference_natorbs=natorbs_ref,
            reference_noons=noons_ref,
            reference_overlap=S_ref,
            return_natorbs=True,
        )
        reference_noons_list.append(noons_ref)
    natorbs_ref=noons_ref=S_ref=None
    for k,x in enumerate(all_x[:]):
        system, natorbs_ref,noons_ref,S_ref = construct_pyscf_system_rhf_canonicalorb(
            molecule=molecule_func(*x),
            basis=basis,
            add_spin=False,
            anti_symmetrize=False,
            reference_natorbs=natorbs_ref,
            reference_noons=noons_ref,
            reference_overlap=S_ref,
            return_natorbs=True,
        )
        if k in desired_samples:
            reference_noons_list.append(noons_ref)
            reference_overlap_list.append(S_ref)
            reference_natorb_list.append(natorbs_ref)
            systems.append(system)
    for k,system in enumerate(systems):
        rccsd = RCCSD(system, verbose=False)
        ground_state_tolerance = 1e-10
        rccsd.compute_ground_state(
            t_kwargs=dict(tol=ground_state_tolerance),
            l_kwargs=dict(tol=ground_state_tolerance),
        )
        t, l = rccsd.get_amplitudes()
        t1s.append(t[0])
        t2s.append(t[1])
        l1s.append(l[0])
        l2s.append(l[1])
        sample_energies.append(system.compute_reference_energy().real+rccsd.compute_energy().real)
        print("Calculated %d"%k)
    return t1s,t2s,l1s,l2s,sample_energies,reference_natorb_list,reference_overlap_list,reference_noons_list
def setUpsamples_naturalOrbitals(all_x,molecule_func,basis,freeze_threshold=0,desired_samples=None):
    """
    Sets up natural orbitals and the corresponding lambda and t-amplitudes for a set of geometries.
    Natural orbitals need to be calculated for each sample_x, even though we only want natural orbitals and cluster
    amplitudes for "desired_samples", because we night a sufficiently small distance between natural orbitals in order
    to categorize them correctly.

    Input:
    all_x (list): The geometries at which the molecule is constructed. Possibly the whole PES of interest
    molecule_func (function->String): A function which returns a string
        corresponding to a molecular geometry as function of parameters from all_x.
    basis (string): The basis set.
    freeze_threshold (float): Cutoff threshold for natural occupation numbers. Cutoff only occurs when
    desired_samples (list): List of integers corresponding to indices in all_x. CCSD equations are only solved for those.

    Returns:
    t1s,t2s,l1s,l2s: CCSD ampltidue and lambda equations
    sample_energies: CCSD Energies for the desired samples
    reference_natorb_list: Coefficient matrices for natural orbitals at desired_samples
    reference_overlap_list: The overlap matrices S at desired_samples
    reference_noons_list: Natural occupation numbers at desired_samples
    """

    if desired_samples is None:
        desired_samples=np.array(np.arange(0,len(all_x)),dtype=int)
    t1s=[]
    t2s=[]
    l1s=[]
    l2s=[]
    sample_energies=[]
    reference_natorb_list=[]
    reference_overlap_list=[]
    reference_noons_list=[]
    systems=[]
    natorbs_ref=noons_ref=S_ref=None
    for k,x in enumerate(all_x[:]):
        system, natorbs_ref,noons_ref,S_ref = construct_pyscf_system_rhf_natorb(
            molecule=molecule_func(*x),
            basis=basis,
            add_spin=False,
            anti_symmetrize=False,
            reference_natorbs=natorbs_ref,
            reference_noons=noons_ref,
            reference_overlap=S_ref,
            return_natorbs=True,
        )
        reference_noons_list.append(noons_ref)
    mindices=[]
    for noons in reference_noons_list:
        mindices.append(np.where(noons>freeze_threshold)[0][-1])
    mindex=np.max(mindices)+1
    natorbs_ref=noons_ref=S_ref=None
    print("Finished first iteration")
    for k,x in enumerate(all_x[:]):
        system, natorbs_ref,noons_ref,S_ref = construct_pyscf_system_rhf_natorb(
            molecule=molecule_func(*x),
            basis=basis,
            add_spin=False,
            anti_symmetrize=False,
            reference_natorbs=natorbs_ref,
            reference_noons=noons_ref,
            reference_overlap=S_ref,
            return_natorbs=True,
            truncation=mindex
        )
        if k in desired_samples:
            reference_noons_list.append(noons_ref)
            reference_overlap_list.append(S_ref)
            reference_natorb_list.append(natorbs_ref)
            systems.append(system)
    print("Finished second iteration")
    for k,system in enumerate(systems):
        rccsd = RCCSD(system, verbose=False)
        ground_state_tolerance = 1e-10
        rccsd.compute_ground_state(
            t_kwargs=dict(tol=ground_state_tolerance),
            l_kwargs=dict(tol=ground_state_tolerance),
        )
        t, l = rccsd.get_amplitudes()
        t1s.append(t[0])
        t2s.append(t[1])
        l1s.append(l[0])
        l2s.append(l[1])
        sample_energies.append(system.compute_reference_energy().real+rccsd.compute_energy().real)
        print("Calculated %d"%k)
    return t1s,t2s,l1s,l2s,sample_energies,reference_natorb_list,reference_overlap_list,reference_noons_list
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
            add_spin=False,
            anti_symmetrize=False,
            reference_state=ref_state,
            mix_states=mix_states,
            weights=None
        )
        rccsd = RCCSD(system, verbose=False)
        ground_state_tolerance = 1e-8
        rccsd.compute_ground_state(
            t_kwargs=dict(tol=ground_state_tolerance),
            l_kwargs=dict(tol=ground_state_tolerance),

        )
        t, l = rccsd.get_amplitudes()
        t1s.append(t[0])
        t2s.append(t[1])
        l1s.append(l[0])
        l2s.append(l[1])
        sample_energies.append(system.compute_reference_energy().real+rccsd.compute_energy().real)
    return t1s,t2s,l1s,l2s,sample_energies
def setUpsamples_givenC(sample_x,molecule_func,basis,givenCs,mix_states=False,type="procrustes",weights=None,guesses=None):
    """
    Sets up lambda and t-amplitudes for a set of geometries where a different reference (givenC) is to be used

    Input:
    sample_x (list): The geometries at which the molecule is constructed.
    molecule_func (function->String): A function which returns a string
        corresponding to a molecular geometry as function of parameters from sample_x.
    basis (string): The basis set.
    givenCs (list of matrices):
        List of reference determinants for the Procrustes algorithm, a different one for each sample_x
    mix_states (bool): if general procrustes orbitals should be used.
    weights: Outdated parameter
    guesses: List of t1, t2, l1 and l2 params to use as starting guess
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
        ref_state=givenCs[k]
        system = construct_pyscf_system_rhf_ref(
        molecule=molecule_func(*x),
        basis=basis,
        add_spin=False,
        anti_symmetrize=False,
        givenC=ref_state,
        verbose=True)
        if guesses is not None and k==0:
            t1_guess=guesses[0][k]
            t2_guess=guesses[1][k]
            l1_guess=guesses[2][k]
            l2_guess=guesses[3][k]
            rccsd = RCCSD(system, verbose=False,start_guess=[t1_guess,t2_guess,l1_guess,l2_guess])
        else:
            t1_guess=t1s[-1]
            t2_guess=t2s[-1]
            l1_guess=l1s[-1]
            l2_guess=l2s[-1]
            rccsd = RCCSD(system, verbose=False,start_guess=[t1_guess,t2_guess,l1_guess,l2_guess])
        ground_state_tolerance = 1e-6
        try:
            rccsd.compute_ground_state(
                t_kwargs=dict(tol=ground_state_tolerance,max_iterations=100),
                l_kwargs=dict(tol=ground_state_tolerance,max_iterations=100),
            )
        except AssertionError:
            #Repeat with lower tolerance and more iterations as a solution is wished for
            t1_guess=t1s[-1]
            t2_guess=t2s[-1]
            l1_guess=l1s[-1]
            l2_guess=l2s[-1]
            rccsd = RCCSD(system, verbose=False,start_guess=[t1_guess,t2_guess,l1_guess,l2_guess])
            rccsd.compute_ground_state(
                t_kwargs=dict(tol=ground_state_tolerance*1000,max_iterations=1000),
                l_kwargs=dict(tol=ground_state_tolerance*1000,max_iterations=1000),
            )
        t, l = rccsd.get_amplitudes()
        t1s.append(t[0])
        t2s.append(t[1])
        l1s.append(l[0])
        l2s.append(l[1])
        sample_energies.append(system.compute_reference_energy().real+rccsd.compute_energy().real)

    return t1s,t2s,l1s,l2s,sample_energies

class EVCSolver():
    """
    Class to solve EVC equations. Contains both WF-CCEVC and AMP-CCEVC functions. Uses restricted determinants and restricted CC theory.

    Methods:
    __init__: Initialize
    solve_CCSD: Return CCSD energies at self.all_x
    solve_WFCCSD: solves WF-CCEVC equations and returns WF-CCEVC energies for given sample amplitudes & lambda equations
    solve_AMP_CCSD: solves AMP-CCEVC equations and returns AMP-CCEVC energies for given sample amplitudes
    """
    def __init__(self,all_x,molecule_func,basis,reference_determinant,t1s,t2s,l1s,l2s,givenC=False,sample_x=None,mix_states=False,natorb_truncation=None):
        self.all_x=all_x
        self.molecule_func=molecule_func
        self.sample_x=sample_x
        self.basis=basis
        self.reference_natorbs=reference_determinant #The reference determiant or a list of different reference determinants. Ignore that it is called "natorbs"
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
    def solve_CCSD_noProcrustes(self,xtol=1e-8):
        """
        Solves the CCSD equations.

        Returns:
        E_CCSD (list): CCSD Energies at all_x.
        """
        self.num_iter=[]
        E_CCSD=[]
        for k,x_alpha in enumerate(self.all_x):
            if isinstance(self.reference_natorbs,list):
                ref_state=self.reference_natorbs[k]
            else:
                ref_state=self.reference_natorbs
            system = construct_pyscf_system_rhf( #Construct a canonical-orbital HF state.
                molecule=self.molecule_func(*x_alpha),
                basis=self.basis,
                add_spin=False,
                anti_symmetrize=False,
            )
            try:
                rccsd = RCCSD(system, verbose=False)
                ground_state_tolerance = xtol
                rccsd.compute_ground_state(t_kwargs=dict(tol=ground_state_tolerance))
                E_CCSD.append(system.compute_reference_energy().real+rccsd.compute_energy().real)
                print("Number iterations: %d"%rccsd.num_iterations)
                self.num_iter.append(rccsd.num_iterations)
            except:
                E_CCSD.append(np.nan)
        return E_CCSD
    def solve_CCSD(self):
        """
        Solves the CCSD equations.

        Returns:
        E_CCSD (list): CCSD Energies at all_x.
        """
        E_CCSD=[]
        self.num_iter=[]
        for k,x_alpha in enumerate(self.all_x):
            if isinstance(self.reference_natorbs,list):
                ref_state=self.reference_natorbs[k]
            else:
                ref_state=self.reference_natorbs
            system = construct_pyscf_system_rhf_ref(
                molecule=self.molecule_func(*x_alpha),
                basis=self.basis,
                add_spin=False,
                anti_symmetrize=False,
                reference_state=ref_state,
                mix_states=self.mix_states,
                truncation=self.natorb_truncation
            )
            try:
                rccsd = RCCSD(system, verbose=False)
                ground_state_tolerance = 1e-8
                rccsd.compute_ground_state(t_kwargs=dict(tol=ground_state_tolerance))
                E_CCSD.append(system.compute_reference_energy().real+rccsd.compute_energy().real)
                print("Number iterations: %d"%rccsd.num_iterations)
                self.num_iter.append(rccsd.num_iterations)
            except:
                E_CCSD.append(np.nan)
        return E_CCSD
    def solve_CCSD_previousgeometry(self,xtol=1e-8):
        """Use the previous geometry as a start guess for CCSD calculations. for the first geometry, no previous geometry is used. (MP2 guess)"""
        E_CCSD=[]
        self.num_iter=[]
        for k,x_alpha in enumerate(self.all_x):
            if isinstance(self.reference_natorbs,list):
                ref_state=self.reference_natorbs[k]
            else:
                ref_state=self.reference_natorbs
            system,C_canonical = construct_pyscf_system_rhf_ref( #With these parameters, canonical orbitals are used!
                molecule=self.molecule_func(*x_alpha),
                basis=self.basis,
                add_spin=False,
                anti_symmetrize=False,
                reference_state=None,
                givenC=None,
                mix_states=self.mix_states,
                truncation=self.natorb_truncation,
                return_C=True
            )
            if k==0:
                rccsd = RCCSD(system, include_singles=True)
            else:
                molecule=pyscf.M(atom=self.molecule_func(*x_alpha),basis=self.basis)
                C_new,Uinv=localize_procrustes(None,C_canonical,None,C_prev,nelec=sum(molecule.nelec), return_R=True) #Unitary to go from canonical orbitals to Procrustes
                #But we want the inverse...
                U=np.conj(Uinv.T) #The unitary operation to go from Procrustes orbitals to canonical orbitals!
                t1_new,t2_new=basischange_clusterOperator(U,start_guess_amplitudes[0],start_guess_amplitudes[1])
                start_guess_amplitudes=[t1_new,t2_new]

                rccsd = RCCSD(system, include_singles=True,start_guess=start_guess_amplitudes)

            ground_state_tolerance = xtol
            rccsd.compute_ground_state(t_kwargs=dict(tol=ground_state_tolerance))
            t, l = rccsd.get_amplitudes()
            start_guess_amplitudes=[t[0],t[1]]
            E_CCSD.append(system.compute_reference_energy().real+rccsd.compute_energy().real)
            print("Number iterations: %d"%rccsd.num_iterations)
            self.num_iter.append(rccsd.num_iterations)
            C_prev=C_canonical
        return E_CCSD
    def solve_CCSD_startguess(self,start_guess_t1_list,start_guess_t2_list, basis_change_from_Procrustes=True,xtol=1e-8):
        """
        Solves the CCSD equations. A start guess for the t amplitudes needs to be provided.
        Optional: If basis change is true,
        a transform of the t1 and the t2 to the canonical orbitals will be performed, assuming that the t1 and t2 amplitudes refer
        to Procrustes orbitals.

        Returns:
        E_CCSD (list): CCSD Energies at all_x.
        """
        E_CCSD=[]
        self.num_iter=[]
        for k,x_alpha in enumerate(self.all_x):
            if isinstance(self.reference_natorbs,list):
                ref_state=self.reference_natorbs[k]
            else:
                ref_state=self.reference_natorbs
            system,C_canonical = construct_pyscf_system_rhf_ref( #With these parameters, canonical orbitals are used!
                molecule=self.molecule_func(*x_alpha),
                basis=self.basis,
                add_spin=False,
                anti_symmetrize=False,
                reference_state=None,
                givenC=None,
                mix_states=self.mix_states,
                truncation=self.natorb_truncation,
                return_C=True
            )
            if basis_change_from_Procrustes:
                molecule=pyscf.M(atom=self.molecule_func(*x_alpha),basis=self.basis)
                C_new,Uinv=localize_procrustes(None,C_canonical,None,ref_state,nelec=sum(molecule.nelec), return_R=True) #Unitary to go from canonical orbitals to Procrustes
                #But we want the inverse...
                U=np.conj(Uinv.T) #The unitary operation to go from Procrustes orbitals to canonical orbitals!
                t1_new,t2_new=basischange_clusterOperator(U,start_guess_t1_list[k],start_guess_t2_list[k])
                start_guess_amplitudes=[t1_new,t2_new]
            else:
                start_guess_amplitudes=[start_guess_t1_list[k],start_guess_t2_list[k]]
            rccsd = RCCSD(system, include_singles=True,start_guess=start_guess_amplitudes)
            ground_state_tolerance = xtol
            rccsd.compute_ground_state(t_kwargs=dict(tol=ground_state_tolerance))
            E_CCSD.append(system.compute_reference_energy().real+rccsd.compute_energy().real)
            print("Number iterations: %d"%rccsd.num_iterations)
            self.num_iter.append(rccsd.num_iterations)
            #Add number of iterations. niter.append(rccsd.num_iter)
            #except:
            #    E_CCSD.append(np.nan)
        return E_CCSD
    def calculate_CCSD_energies_from_guess(self,start_guess_t1_list,start_guess_t2_list, basis_change_from_Procrustes=True,xtol=1e-8):
        E_CCSD=[]
        for k,x_alpha in enumerate(self.all_x):
            if isinstance(self.reference_natorbs,list):
                ref_state=self.reference_natorbs[k]
            else:
                ref_state=self.reference_natorbs
            system,C_canonical = construct_pyscf_system_rhf_ref( #With these parameters, canonical orbitals are used!
                molecule=self.molecule_func(*x_alpha),
                basis=self.basis,
                add_spin=False,
                anti_symmetrize=False,
                reference_state=None,
                givenC=None,
                mix_states=self.mix_states,
                truncation=self.natorb_truncation,
                return_C=True
            )
            if basis_change_from_Procrustes:
                molecule=pyscf.M(atom=self.molecule_func(*x_alpha),basis=self.basis)
                C_new,Uinv=localize_procrustes(None,C_canonical,None,ref_state,nelec=sum(molecule.nelec), return_R=True) #Unitary to go from canonical orbitals to Procrustes
                #But we want the inverse...
                U=np.conj(Uinv.T) #The unitary operation to go from Procrustes orbitals to canonical orbitals!
                t1_new,t2_new=basischange_clusterOperator(U,start_guess_t1_list[k],start_guess_t2_list[k])
                start_guess_amplitudes=[t1_new,t2_new]
            else:
                start_guess_amplitudes=[start_guess_t1_list[k],start_guess_t2_list[k]]
            rccsd = RCCSD(system, include_singles=True,start_guess=start_guess_amplitudes)
            reference_energy=system.compute_reference_energy().real
            CCSD_correction=rccsd.compute_energy().real
            E_CCSD.append(reference_energy+CCSD_correction)
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
                    add_spin=False,
                    anti_symmetrize=False,
                    reference_state=ref_state,
                    mix_states=self.mix_states,
                    truncation=self.natorb_truncation
                )
            else:
                system = construct_pyscf_system_rhf_ref(
                    molecule=self.molecule_func(*x_alpha),
                    basis=self.basis,
                    add_spin=False,
                    anti_symmetrize=False,
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

    def solve_AMP_CCSD(self,occs=1,virts=0.5,xtol=1e-5,maxfev=60, start_guess_list=None):
        """
        Solves the AMP_CCSD equations.

        Input:
        occs (float): The percentage (if below 1) or number (if above 1) of occupied orbitals to include in amplitude calculations
        virts (float): The percentage (if below 1) or number (if above 1) of virtual orbitals to include in amplitude calculations
        xtol (float): Convergence criterion for maximum amplitude error
        maxfev (int): Maximal number of Newton's method iterations
        start_guess (list): List of lists with starting parameters [[c_1,\dots,c_L]_1, [c_1,\dots,c_L]_2, \dots]
        Returns:
        energy (list): AMP-CCEVC Energies at all_x.
        """
        energy=[]
        t1_copy=self.t1s #Reset after a run of AMP_CCEVC such that WF-CCEVC can be run afterwards
        t2_copy=self.t2s
        t1s_orth,t2s_orth,coefs=orthonormalize_ts(self.t1s,self.t2s) # Use orthonormal t1 and t2 amplitudes
        if self.coefs is None:
            self.coefs=coefs
        self.t1s=t1s_orth #Use the orthogonalized forms of t1s and t2s
        self.t2s=t2s_orth
        t2=np.zeros(self.t2s[0].shape)
        for i in range(len(self.t1s)):
            t2+=self.t2s[i]
        t1_v_ordering=contract=np.einsum("abij->a",t2**2)
        t1_o_ordering=contract=np.einsum("abij->i",t2**2)
        important_o=np.argsort(t1_o_ordering)[::-1]
        important_v=np.argsort(t1_v_ordering)[::-1]
        chosen_t=np.zeros(self.t2s[0].shape)
        if occs is None:
            occs_local=self.t2s[0].shape[2]
        elif occs<1.1:
            occs_local=int(self.t2s[0].shape[0]*occs)
        else:
            occs_local=occs
        if virts is None:
            virts_local=self.t2s[0].shape[0]//2
        elif virts<1.1:
            virts_local=int(self.t2s[0].shape[0]*virts)
        else:
            virts_local=virts
        self.used_o=np.sort(important_o[:occs_local])
        self.used_v=np.sort(important_v[:virts_local])
        chosen_t[np.ix_(self.used_v,self.used_v,self.used_o,self.used_o)]=1
        self.picks=chosen_t.reshape(self.t2s[0].shape)
        self.picks=(self.picks*(-1)+1).astype(bool)

        self.nos=important_o[:occs_local]
        self.nvs=important_v[:virts_local]
        self.num_iterations=[]
        self.times=[]
        self.projection_errors=[]
        projection_errors=[]
        self.t1s_final=[] #List of the t1 solutions for each x
        self.t2s_final=[] #list of the t2 solutions for each x
        for k,x_alpha in enumerate(self.all_x):
            if isinstance(self.reference_natorbs,list):
                ref_state=self.reference_natorbs[k]
            else:
                ref_state=self.reference_natorbs
            system = construct_pyscf_system_rhf_ref(
                molecule=self.molecule_func(*x_alpha),
                basis=self.basis,
                add_spin=False,
                anti_symmetrize=False,
                reference_state=ref_state,
                mix_states=self.mix_states,
                truncation=self.natorb_truncation
            )
            f = system.construct_fock_matrix(system.h, system.u)
            ESCF=system.compute_reference_energy().real
            self._system_jacobian(system)
            closest_sample_x=np.argmin(np.linalg.norm(np.array(self.sample_x)-x_alpha,axis=1))
            try:
                start_guess=start_guess_list[k]
            except TypeError: #Not a list
                start_guess=self.coefs[:,closest_sample_x]

            except IndexError: #List to short
                start_guess=self.coefs[:,closest_sample_x]
            self.times_temp=[]

            sol=self._own_root_diis(start_guess,args=[system],options={"xtol":xtol,"maxfev":maxfev})
            final=sol.x
            self.num_iterations.append(sol.nfev)
            self.times.append(np.sum(np.array(self.times_temp)))
            t1=np.zeros(self.t1s[0].shape)
            t2=np.zeros(self.t2s[0].shape)
            for i in range(len(self.t1s)):
                t1+=final[i]*self.t1s[i] #Starting guess
                t2+=final[i]*self.t2s[i] #Starting guess
            self.t1s_final.append(t1)
            self.t2s_final.append(t2)
            t1_error = rhs_t.compute_t_1_amplitudes(f, system.u, t1, t2, system.o, system.v, np)
            t2_error = rhs_t.compute_t_2_amplitudes(f, system.u, t1, t2, system.o, system.v, np)
            max_proj_error=np.max((np.max(abs(t1_error)),np.max(abs(t2_error)) ))
            self.projection_errors.append(max_proj_error)
            newEn=rhs_e.compute_rccsd_ground_state_energy(f, system.u, t1, t2, system.o, system.v, np)+ESCF
            if sol.success==False:
                energy.append(np.nan)
            else:
                energy.append(newEn)
        self.t1s=t1_copy
        self.t2s=t2_copy
        return energy
    def _construct_H_S(self,system):
        """
        Constructs H and S matrix for WF-CCEVC.
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
            exp_energy=rhs_e.compute_rccsd_ground_state_energy(f, system.u, self.t1s[i], self.t2s[i], system.o, system.v, np)+RHF_energy
            for j in range(len(self.t1s)):
                X1=self.t1s[i]-self.t1s[j]
                X2=self.t2s[i]-self.t2s[j]
                overlap=0
                overlap+=1+contract("ia,ai->",self.l1s[j],X1)
                overlap+=0.5*contract("ijab,ai,bj->",self.l2s[j],X1,X1)
                overlap+=0.5*contract("ijab,abij->",self.l2s[j],X2) #Extra *2 for RCCSD. Not derived explicitely, but chosen so that it matches with GCCSD!
                S[i,j]=overlap
                extra=contract("ia,ai->",self.l1s[j],t1_error)+contract("ijab,ai,bj->",self.l2s[j],X1,t1_error)
                extra+=0.5*contract("ijab,abij->",self.l2s[j],t2_error) #Extra *2 for RCCSD. Not derived explicitely, but chosen so that it matches with GCCSD!
                H[i,j]=overlap*exp_energy
                H[i,j]+=extra
        return H,S
    def _system_jacobian(self,system):
        """
        Construct quasi-newton jacobian for a given geometry.
        """
        f = system.construct_fock_matrix(system.h, system.u)
        no=system.n
        nv=system.l-system.n
        t1_Jac=np.zeros((nv,no))
        t2_Jac=np.zeros((nv,nv,no,no))
        for a in range(nv):
            for i in range(no):
                t1_Jac[a,i]=f[a+no,a+no]-f[i,i]
        for a in range(nv):
            for b in range(nv):
                for i in range(no):
                    for j in range(no):
                        t2_Jac[a,b,i,j]=f[a+no,a+no]-f[i,i]+f[b+no,b+no]-f[j,j] #This is really crappy, as the diagonal approximation does not hold. Though it works!!
        self.t1_Jac=t1_Jac
        self.t2_Jac=t2_Jac
    def _error_function(self,params,system):
        """
        Projection error given a set of pareters [c_1,...,c_L]

        Input:
        params (list): The parameters [c_1,...,c_L] for which to caluclate the error
        system: The molecule with information

        Returns:
            List: Projection errors
        """
        t1=np.zeros(self.t1s[0].shape)
        t2=np.zeros(self.t2s[0].shape)

        for i in range(len(self.t1s)):
            t1+=params[i]*self.t1s[i] #Starting guess
            t2+=params[i]*self.t2s[i] #Starting guess

        f = system.construct_fock_matrix(system.h, system.u)
        start=time.time()
        t1_error = rhs_t.compute_t_1_amplitudes_REDUCED_new(f, system.u, t1, t2, system.o, system.v, np,self.picks,self.nos,self.nvs)
        t2_error = rhs_t.compute_t_2_amplitudes_REDUCED_new(f, system.u, t1, t2, system.o, system.v, np,self.picks,self.nos,self.nvs)
        end=time.time()
        self.times_temp.append(end-start)
        ts=[np.concatenate((self.t1s[i],self.t2s[i]),axis=None) for i in range(len(self.t1s))]
        t_error=np.concatenate((t1_error,t2_error),axis=None)
        projection_errors=np.zeros(len(self.t1s))
        t1_error_flattened=t1_error.flatten()
        t2_error_flattened=t2_error.flatten()
        for i in range(len(projection_errors)):
            projection_errors[i]+=t1_error_flattened@self.t1s[i].flatten()
            projection_errors[i]+=t2_error_flattened@self.t2s[i].flatten()
        return projection_errors
    def _jacobian_function(self,params,system):
        """
        quasi-newton jacobian for a given geometry as expressed in terms of params [c_1,\dots,c_L]
        """
        t1=np.zeros(self.t1s[0].shape)
        t2=np.zeros(self.t2s[0].shape)
        Ts=[]
        for i in range(len(self.t1s)):
            t1+=params[i]*self.t1s[i] #Starting guess
            t2+=params[i]*self.t2s[i] #Starting guess
            #Ts.append(t2s[i][picks].flatten())
            Ts.append(self.t2s[i][np.ix_(self.used_v,self.used_v,self.used_o,self.used_o)].flatten())

        jacobian_matrix=np.zeros((len(params),len(params)))
        for i in range(len(params)):
            for j in range(i,len(params)):
                jacobian_matrix[j,i]+=contract("k,k,k->",self.t1s[i][np.ix_(self.used_v,self.used_o)].flatten(),self.t1s[j][np.ix_(self.used_v,self.used_o)].flatten(),self.t1_Jac[np.ix_(self.used_v,self.used_o)].flatten())
                jacobian_matrix[j,i]+=contract("k,k,k->",Ts[i],Ts[j],self.t2_Jac[np.ix_(self.used_v,self.used_v,self.used_o,self.used_o)].flatten())
                jacobian_matrix[i,j]=jacobian_matrix[j,i]
        return jacobian_matrix
    def _own_root_diis(self,start_guess,args,options={"xtol":1e-3,"maxfev":25},diis_start=2,diis_dim=5):
        """Root finding algorithm based on quasi-newton and DIIS.

        Input:
        start_guess (list): Start guess for coefficients
        args [list]: A list containing the system
        options (dict): Tolerance (xtol) and number of maximum evaluations
        diis_start: How many iterations to do before starting diis_start
        diis_dim: Dimensionality of DIIS subspace

        Returns:
        Sucess object with final solution, number of iterations, and wether convergence was reached
        """
        guess=start_guess
        xtol=options["xtol"]
        maxfev=options["maxfev"]
        iter=0
        errors=[]
        amplitudes=[]
        updates=[]
        error=10*xtol #Placeholder to enter while loop

        #Calculate diis_start iterations without DIIS
        while np.max(np.abs(error))>xtol and iter<diis_start:
            jacobian=self._jacobian_function(guess,*args)
            error=self._error_function(guess,*args)
            update=-np.linalg.inv(jacobian)@error
            guess=guess+update
            errors.append(error)
            updates.append(update)
            amplitudes.append(guess)
            iter+=1

        #DIIS algorithm
        while np.max(np.abs(error))>xtol and iter<maxfev:

            #Calculate DIIS B-matrix from Jacobian guess update
            B_matrix=np.zeros((len(updates)+1,len(updates)+1))
            for i,ei in enumerate(updates):
                for j, ej in enumerate(updates):
                    B_matrix[i,j]=np.dot(ei,ej)
            for i in range(len(updates)+1):
                B_matrix[i,-1]=B_matrix[-1,i]=-1
            B_matrix[-1,-1]=0
            sol=np.zeros(len(updates)+1)
            sol[-1]=-1
            input=np.zeros(len(updates[0]))
            try:
                weights=np.linalg.solve(B_matrix,sol)
                for i,w in enumerate(weights[:-1]):
                    input+=w*amplitudes[i] #Calculate new approximate ampltiude guess vector
                    #errsum+=w*updates[i]
            except np.linalg.LinAlgError: #If DIIS matrix is singular, use most recent quasi-newton step
                input=guess
            #errsum=np.zeros(len(updates[0]))


            jacobian=self._jacobian_function(input,*args)
            error=self._error_function(input,*args)
            update=-np.linalg.inv(jacobian)@error #Calculate update vector
            guess=input+update
            errors.append(error)
            updates.append(update)
            amplitudes.append(guess)
            if len(errors)>=diis_dim: #Reduce DIIS space to dimensionality threshold
                errors.pop(0)
                amplitudes.pop(0)
                updates.pop(0)
            iter+=1
        success=iter<maxfev
        print("Num iter: %d"%iter)
        return sucess(guess,success,iter)


def get_reference_determinant(molecule_func,refx,basis,charge=0):
    mol = gto.Mole()
    mol.unit = "bohr"
    mol.charge = charge
    mol.cart = False
    mol.build(atom=molecule_func(*refx), basis=basis)
    hf = scf.RHF(mol)
    hf.kernel()
    return np.asarray(hf.mo_coeff)
def get_natural_orbitals(molecule_func,xvals,basis,natorbs_ref=None,noons_ref=None,Sref=None):
    """
    Obtain the natural orbitals in analytic order for a molecule at a given set of geometries (xvals).
    If references are given, the reference decides the ordering of the NOs.
    """
    natorbs_list=[]
    overlaps_list=[]
    noons_list=[]
    for x in xvals:
        mol = gto.Mole()
        mol.unit = "bohr"
        mol.charge = 0
        mol.cart = False
        mol.build(atom=molecule_func(*x), basis=basis)
        hf = scf.RHF(mol)
        hf.kernel()
        mymp=mp.RMP2(hf).run()
        noons,natorbs=mcscf.addons.make_natural_orbitals(mymp)
        S=mol.intor("int1e_ovlp")
        if natorbs_ref is not None:
            noons_ref,natorbs_ref=similiarize_natural_orbitals(noons_ref,natorbs_ref,noons,natorbs,mol.nelec,S,Sref)
        else:
            noons_ref=noons; natorbs_ref=natorbs; S_ref=S
        natorbs_list.append(natorbs_ref)
        Sref=S
        overlaps_list.append(S)
        noons_list.append(noons)
    return natorbs_list,noons_list,overlaps_list
def get_canonical_orbitals(molecule_func,xvals,basis,natorbs_ref=None,noons_ref=None,Sref=None):
    """
    Obtain the canonical orbitals in analytic order for a molecule at a given set of geometries (xvals).
    If references are given, the reference decides the ordering of the NOs.
    """
    natorbs_list=[]
    overlaps_list=[]
    noons_list=[]
    for x in xvals:
        mol = gto.Mole()
        mol.unit = "bohr"
        mol.charge = 0
        mol.cart = False
        mol.build(atom=molecule_func(*x), basis=basis)
        hf = scf.RHF(mol)
        hf.kernel()
        noons=hf.mo_energy
        natorbs=hf.mo_coeff
        S=mol.intor("int1e_ovlp")
        if natorbs_ref is not None:
            noons_ref,natorbs_ref=similiarize_canonical_orbitals(noons_ref,natorbs_ref,noons,natorbs,mol.nelec,S,Sref)
        else:
            noons_ref=noons; natorbs_ref=natorbs; S_ref=S
        natorbs_list.append(natorbs_ref)
        Sref=S
        overlaps_list.append(S)
        noons_list.append(noons)
    return natorbs_list,noons_list,overlaps_list

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

    E_AMP_full=evcsolver.solve_AMP_CCSD(occs=1,virts=1)
    E_AMP_red=evcsolver.solve_AMP_CCSD(occs=1,virts=0.5)
    E_CCSDx=evcsolver.solve_CCSD()
    E_WF=evcsolver.solve_WFCCEVC()
    plt.plot(geom_alphas1,E_CCSDx,label="CCSD")
    plt.plot(geom_alphas1,E_WF,label="WF-CCEVC")
    plt.plot(geom_alphas1,E_AMP_full,label=r"AMP-CCEVC (50$\%$)")
    plt.plot(geom_alphas1,E_AMP_red,label=r"AMP-CCEVC")
    plt.tight_layout()
    plt.legend()
    plt.show()
