from qs_ref import *
import basis_set_exchange as bse
from coupled_cluster.rccsd import RCCSD, TDRCCSD
from coupled_cluster.rccsd import rhs_t
from coupled_cluster.rccsd import energies as rhs_e
from opt_einsum import contract
from full_cc import orthonormalize_ts, orthonormalize_ts_choice, orthonormalize_ts_pca
from scipy.optimize import minimize, root,newton
import time
class sucess():
    """Mini helper class that resembles scipy's OptimizeResult."""
    def __init__(self,x,success,nfev):
        self.x=x
        self.success=success
        self.nfev=nfev

def setUpsamples_naturalOrbitals_givenNatorbs(sample_x,molecule_func,basis,freeze_threshold,natorbs,noons,Ss,mindex):
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
    for k,x in enumerate(sample_x):
        mol=gto.Mole()
        mol.atom=molecule_func(*x)
        mol.basis = basis
        mol.unit= "Bohr"
        mol.build()
        system, natorbs_ref,noons_ref,S_ref = construct_pyscf_system_rhf_natorb(
            molecule=molecule_func(*x),
            basis=basis,
            add_spin=False,
            anti_symmetrize=False,
            reference_natorbs=natorbs[k],
            reference_noons=noons[k],
            reference_overlap=Ss[k],
            return_natorbs=True,
            truncation=mindex
        )
        reference_noons_list.append(noons_ref)
        reference_overlap_list.append(S_ref)
        reference_natorb_list.append(natorbs_ref)
        systems.append(system)
    for system in systems:
        rccsd = RCCSD(system, verbose=False)
        ground_state_tolerance = 1e-7
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

def setUpsamples_naturalOrbitals(sample_x,molecule_func,basis,freeze_threshold=0,desired_samples=None):
    if desired_samples is None:
        desired_samples=np.array(np.arange(0,len(sample_x)),dtype=int)
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
    for k,x in enumerate(sample_x[:]):
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
    (print(mindex,len(noons)))
    print("Finished first iteration")
    for k,x in enumerate(sample_x[:]):
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
        ground_state_tolerance = 1e-7
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
            reference_state=rhf_mo_ref,
            mix_states=mix_states,
            weights=None
        )
        rccsd = RCCSD(system, verbose=False)
        ground_state_tolerance = 1e-7
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

class EVCSolver():
    def __init__(self,x_alphas,molecule_func,basis,reference_natorbs,t1s,t2s,l1s,l2s,sample_x=None,mix_states=False,natorb_truncation=None):
        self.x_alphas=x_alphas
        self.molecule_func=molecule_func
        self.sample_x=sample_x
        self.basis=basis
        self.reference_natorbs=reference_natorbs
        self.t1s=t1s
        self.t2s=t2s
        self.l1s=l1s
        self.l2s=l2s
        self.mix_states=mix_states
        self.natorb_truncation=natorb_truncation
        self.num_params=np.prod(t1s[0].shape)+np.prod(t2s[0].shape)
    def construct_H_S(self,system):
        RHF_energy=system.compute_reference_energy().real
        H=np.zeros((len(self.t1s),len(self.t1s)))
        S=np.zeros((len(self.t1s),len(self.t1s)))
        for i in range(len(self.t1s)):
            f = system.construct_fock_matrix(system.h, system.u)
            t1_error = rhs_t.compute_t_1_amplitudes(f, system.u, self.t1s[i], self.t2s[i], system.o, system.v, np)
            t2_error = rhs_t.compute_t_2_amplitudes(f, system.u, self.t1s[i], self.t2s[i], system.o, system.v, np)
            exp_energy=rhs_e.compute_rccsd_ground_state_energy(f, system.u, self.t1s[i], self.t2s[i], system.o, system.v, np)+RHF_energy
            for j in range(len(self.t1s)):
                X1=self.t1s[i]-self.t1s[j]
                X2=self.t2s[i]-self.t2s[j]
                overlap=1+contract("ia,ai->",self.l1s[j],X1)+0.5*contract("ijab,ai,bj->",self.l2s[j],X1,X1)+0.25*contract("ijab,abij->",self.l2s[j],X2)
                S[i,j]=overlap

                H[i,j]=overlap*exp_energy
                extra=contract("ia,ai->",self.l1s[j],t1_error)+contract("ijab,ai,bj->",self.l2s[j],X1,t1_error)+0.25*contract("ijab,abij->",self.l2s[j],t2_error)
                H[i,j]=H[i,j]+extra
        return H,S
    def solve_WFCCEVC(self):
        E_WFCCEVC=[]
        for k,x_alpha in enumerate(self.x_alphas):
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
            H,S=self.construct_H_S(system)
            E_WFCCEVC.append(guptri_Eigenvalue(H,S))
        return E_WFCCEVC
    def solve_CCSD(self):
        E_CCSD=[]
        for k,x_alpha in enumerate(self.x_alphas):
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
                print("Starting to calculate ground state energy")
                rccsd = RCCSD(system, verbose=False)
                ground_state_tolerance = 1e-7
                rccsd.compute_ground_state(t_kwargs=dict(tol=ground_state_tolerance))
                E_CCSD.append(system.compute_reference_energy().real+rccsd.compute_energy().real)
            except:
                E_CCSD.append(np.nan)
        return E_CCSD
    def solve_AMP_CCSD(self,occs=1,virts=0.5):
        energy=[]
        start_guess=np.full(len(self.t1s),1/len(self.t1s))
        t1s_orth,t2s_orth,coefs=orthonormalize_ts(self.t1s,self.t2s)
        self.t1s=t1s_orth
        self.t2s=t2s_orth
        t2=np.zeros(self.t2s[0].shape)
        for i in range(len(self.t1s)):
            t2+=self.t2s[i]
        t1_v_ordering=contract=np.einsum("abij->a",t2**2)
        t1_o_ordering=contract=np.einsum("abij->i",t2**2)
        important_o=np.argsort(t1_o_ordering)[::-1]
        important_v=np.argsort(t1_v_ordering)[::-1]
        pickerino=np.zeros(self.t2s[0].shape)
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
        pickerino[np.ix_(important_v[:virts_local],important_v[:virts_local],important_o[:occs_local],important_o[:occs_local])]=1
        self.picks=pickerino.reshape(self.t2s[0].shape)
        self.picks=(self.picks*(-1)+1).astype(bool)

        self.cutoff=occs_local**2*virts_local**2
        self.nos=important_o[:occs_local]
        self.nvs=important_v[:virts_local]
        self.num_iterations=[]
        self.times=[]

        for k,x_alpha in enumerate(self.x_alphas):
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
            closest_sample_x=np.argmin(abs(np.array(self.sample_x)-x_alpha))
            start_guess=coefs[:,closest_sample_x]
            self.times_temp=[]
            sol=self._own_root_diis(start_guess,args=[system],options={"xtol":1e-3,"maxfev":100})
            final=sol.x
            print(final)
            self.num_iterations.append(sol.nfev)
            self.times.append(self.times_temp)
            t1=np.zeros(self.t1s[0].shape)
            t2=np.zeros(self.t2s[0].shape)
            for i in range(len(self.t1s)):
                t1+=final[i]*self.t1s[i] #Starting guess
                t2+=final[i]*self.t2s[i] #Starting guess
            t1_error = rhs_t.compute_t_1_amplitudes(f, system.u, t1, t2, system.o, system.v, np)
            t2_error = rhs_t.compute_t_2_amplitudes(f, system.u, t1, t2, system.o, system.v, np)
            newEn=rhs_e.compute_rccsd_ground_state_energy(f, system.u, t1, t2, system.o, system.v, np)+ESCF
            energy.append(newEn)
        return energy
    def _system_jacobian(self,system):
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
                        t2_Jac[a,b,i,j]=f[a+no,a+no]-f[i,i]+f[b+no,b+no]-f[j,j]
        self.t1_Jac=t1_Jac
        self.t2_Jac=t2_Jac
    def _error_function(self,params,system):
        t1=np.zeros(self.t1s[0].shape)
        t2=np.zeros(self.t2s[0].shape)

        for i in range(len(self.t1s)):
            t1+=params[i]*self.t1s[i] #Starting guess
            t2+=params[i]*self.t2s[i] #Starting guess

        f = system.construct_fock_matrix(system.h, system.u)
        #t1_error = rhs_t.compute_t_1_amplitudes(f, system.u, t1, t2, system.o, system.v, np)
        start=time.time()
        t1_error = rhs_t.compute_t_1_amplitudes_REDUCED_new(f, system.u, t1, t2, system.o, system.v, np,self.picks,self.nos,self.nvs) #Original idea
        t2_error = rhs_t.compute_t_2_amplitudes_REDUCED_new(f, system.u, t1, t2, system.o, system.v, np,self.picks,self.nos,self.nvs) #Original idea
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
        t1=np.zeros(self.t1s[0].shape)
        t2=np.zeros(self.t2s[0].shape)
        Ts=[]
        for i in range(len(self.t1s)):
            t1+=params[i]*self.t1s[i] #Starting guess
            t2+=params[i]*self.t2s[i] #Starting guess
            #Ts.append(t2s[i][picks].flatten())
            Ts.append(self.t2s[i].flatten())
        jacobian_matrix=np.zeros((len(params),len(params)))
        for i in range(len(params)):
            for j in range(i,len(params)):
                jacobian_matrix[j,i]+=contract("k,k,k->",self.t1s[i].flatten(),self.t1s[j].flatten(),self.t1_Jac.flatten())
                #jacobian_matrix[j,i]+=contract("k,k,k->",Ts[i],Ts[j],t2_Jac[picks].flatten())
                jacobian_matrix[j,i]+=contract("k,k,k->",Ts[i],Ts[j],self.t2_Jac.flatten())
                jacobian_matrix[i,j]=jacobian_matrix[j,i]
        return jacobian_matrix
    def _own_root_diis(self,start_guess,args,options={"xtol":1e-3,"maxfev":25},diis_start=2,diis_dim=5):
        guess=start_guess
        xtol=options["xtol"]
        maxfev=options["maxfev"]
        iter=0
        errors=[]
        amplitudes=[]
        updates=[]
        error=10*xtol #Placeholder to enter while loop
        while np.max(np.abs(error))>xtol and iter<diis_start:
            jacobian=self._jacobian_function(guess,*args)

            error=self._error_function(guess,*args)
            update=-np.linalg.inv(jacobian)@error
            guess=guess+update
            print(error)
            errors.append(error)
            updates.append(update)
            amplitudes.append(guess)
            iter+=1
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
            weights=np.linalg.solve(B_matrix,sol)
            input=np.zeros(len(updates[0]))
            #errsum=np.zeros(len(updates[0]))
            for i,w in enumerate(weights[:-1]):
                input+=w*amplitudes[i] #Calculate new approximate ampltiude guess vector
                #errsum+=w*updates[i]

            jacobian=self._jacobian_function(input,*args)
            error=self._error_function(input,*args)
            print(error)
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

if __name__=="__main__":
    basis = '6-31G*'
    basis_set = bse.get_basis(basis, fmt='nwchem')
    charge = 0
    #molecule =lambda arr: "Be 0.0 0.0 0.0; H 0.0 0.0 %f; H 0.0 0.0 -%f"%(arr,arr)
    #molecule=lambda x:  "H 0 0 %f; H 0 0 -%f; Be 0 0 0"%(x,x)
    molecule=lambda x:  "H 0 0 %f; F 0 0 0"%(x)
    refx=[2]
    print(molecule(*refx))
    reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
    sample_geom1=np.linspace(2,4,4)
    #sample_geom1=[2.5,3.0,6.0]
    sample_geom=[[x] for x in sample_geom1]
    sample_geom1=np.array(sample_geom).flatten()
    geom_alphas1=np.linspace(1.2,5.0,20)
    geom_alphas=[[x] for x in geom_alphas1]

    t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis_set,reference_determinant,mix_states=False,type="procrustes")
    evcsolver=EVCSolver(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,sample_x=sample_geom,mix_states=False,natorb_truncation=None)
    E_WF=evcsolver.solve_WFCCEVC()
    E_AMP_full=evcsolver.solve_AMP_CCSD(occs=1,virts=1)
    E_AMP_red=evcsolver.solve_AMP_CCSD(occs=1,virts=0.5)
    E_CCSDx=evcsolver.solve_CCSD()
    print(E_approx,E_CCSDx)
    axes[i][j].plot(geom_alphas1,E_CCSDx,label="CCSD")
    axes[i][j].plot(geom_alphas1,E_approx,label="WF-CCEVC")
    axes[i][j].plot(geom_alphas1,E_AMP_full,label=r"AMP-CCEVC (50$\%$)")
    axes[i][j].plot(geom_alphas1,E_AMP_red,label=r"AMP-CCEVC")
    #plt.plot(geom_alphas1,energy_simen_random,label="CCSD AMP 2")
    plt.legend()
    plt.show()
