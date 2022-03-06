from qs_ref import *
import basis_set_exchange as bse
from coupled_cluster.rccsd import RCCSD, TDRCCSD
from coupled_cluster.rccsd import rhs_t
from coupled_cluster.rccsd import energies as rhs_e
from opt_einsum import contract
from full_cc import orthonormalize_ts, orthonormalize_ts_choice
from scipy.optimize import minimize, root,newton

def setUpsamples_naturalOrbitals(sample_x,molecule_func,basis,freeze_threshold):
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
    for x in sample_x[:]:
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
    mindex=np.max(mindices)
    natorbs_ref=noons_ref=S_ref=None
    for x in sample_x[:]:
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
        reference_noons_list.append(noons_ref)
        reference_overlap_list.append(S_ref)
        reference_natorb_list.append(natorbs_ref)
        systems.append(system)
    for system in systems:
        rccsd = RCCSD(system, verbose=False)
        ground_state_tolerance = 5e-15
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
    return t1s,t2s,l1s,l2s,sample_energies,reference_natorb_list,reference_overlap_list,reference_noons_list
def setUpsamples(sample_x,molecule_func,basis,rhf_mo_ref,mix_states=False,type="procrustes",weights=None):
    t1s=[]
    t2s=[]
    l1s=[]
    l2s=[]
    sample_energies=[]
    for x in sample_x:
        system = construct_pyscf_system_rhf_ref(
            molecule=molecule_func(*x),
            basis=basis,
            add_spin=False,
            anti_symmetrize=False,
            reference_state=rhf_mo_ref,
            mix_states=mix_states,
            weights=weights
        )
        rccsd = RCCSD(system, verbose=False)
        ground_state_tolerance = 5e-15
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
def solve_evc(x_alphas,molecule_func,basis,reference_natorbs,t1s,t2s,l1s,l2s,mix_states=False,run_cc=True,cc_approx=True,type=None,tol=3e-8,weights=None,truncation=1000000):
    """
    x_alphas: The sample geometries inpupt
    molecule: The function which returns the molecule
    basis: The basis set used
    reference_natorbs: The reference state OR the natorbs. Either a matrix or a list
    t1s, t2s, l1s, l2s: The set of CC-coefficients to build the system with
    run_cc: Wether to solve the CC equations as well or not.
    cc_approx: Wether to get an approximative CC state or not
    """
    E_CCSD=[]
    E_approx=[]
    E_ownmethod=[]
    E_diffguess=[]
    E_RHF=[]
    for k,x_alpha in enumerate(x_alphas):
        if isinstance(reference_natorbs,list):
            ref_state=reference_natorbs[k]
        else:
            ref_state=reference_natorbs
        system = construct_pyscf_system_rhf_ref(
            molecule=molecule_func(*x_alpha),
            basis=basis,
            add_spin=False,
            anti_symmetrize=False,
            reference_state=ref_state,
            mix_states=mix_states,
            weights=weights,
            truncation=truncation

        )
        ESCF=system.compute_reference_energy().real
        if run_cc:
            try:
                rccsd = RCCSD(system, verbose=False)
                ground_state_tolerance = 1e-10
                rccsd.compute_ground_state(
                    t_kwargs=dict(tol=ground_state_tolerance)                )
                E_CCSD.append(ESCF+rccsd.compute_energy().real)
            except:
                E_CCSD.append(np.nan)
        H=np.zeros((len(t1s),len(t1s)))
        S=np.zeros((len(t1s),len(t1s)))
        for i, xi in enumerate(t1s):
            f = system.construct_fock_matrix(system.h, system.u)
            t1_error = rhs_t.compute_t_1_amplitudes(f, system.u, t1s[i], t2s[i], system.o, system.v, np)
            t2_error = rhs_t.compute_t_2_amplitudes(f, system.u, t1s[i], t2s[i], system.o, system.v, np)
            exp_energy=rhs_e.compute_rccsd_ground_state_energy(f, system.u, t1s[i], t2s[i], system.o, system.v, np)+ESCF
            for j, xj in enumerate(t1s):
                X1=t1s[i]-t1s[j]
                X2=t2s[i]-t2s[j]
                overlap=1+contract("ia,ai->",l1s[j],X1)+0.5*contract("ijab,ai,bj->",l2s[j],X1,X1)+0.25*contract("ijab,abij->",l2s[j],X2)
                S[i,j]=overlap

                H[i,j]=overlap*exp_energy
                extra=contract("ia,ai->",l1s[j],t1_error)+contract("ijab,ai,bj->",l2s[j],X1,t1_error)+0.25*contract("ijab,abij->",l2s[j],t2_error)
                H[i,j]=H[i,j]+extra
        """
        e,cl,c=eig(scipy.linalg.pinv(S,atol=tol)@H,left=True)

        idx = np.real(e).argsort()
        e = e[idx]
        c = c[:,idx]
        cl = cl[:,idx]
        E_approx.append(np.real(e[0]))
        """
        #E_approx.append(schur_lowestEigenValue(H,S))
        E_approx.append(guptri_Eigenvalue(H,S))
        print (E_approx[-1],E_CCSD[-1])
        if cc_approx:
            t1own,t2own=best_CC_fit(t1s,t2s,np.real(c[:,0])/np.sum(np.real(c[:,0])))
            own_energy=gccsolver.ccsdenergy(t1own,t2own)+ESCF
            E_ownmethod.append(own_energy)
    return E_CCSD,E_approx,E_diffguess,E_RHF,E_ownmethod
def solve_evc2(x_alphas,molecule_func,basis,reference_natorbs,t1s,t2s,l1s,l2s,mix_states=False,random_picks=0,type="procrustes",optimal=True,weights=None,truncation=1000000):
    for i in range(len(t1s)):
        t1s[i]=np.real(t1s[i])
        t2s[i]=np.real(t2s[i])
        l1s[i]=np.real(l1s[i])
        l2s[i]=np.real(l2s[i])
    def system_jacobian(system):
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
        return t1_Jac,t2_Jac

    def error_function(params,system,t1s,t2s,t1_Jac,t2_Jac,picks):
        def erf(params):
            t1=np.zeros(t1s[0].shape)
            t2=np.zeros(t2s[0].shape)

            for i in range(len(t1s)):
                t1+=params[i]*t1s[i] #Starting guess
                t2+=params[i]*t2s[i] #Starting guess

            f = system.construct_fock_matrix(system.h, system.u)

            removed_t1=np.zeros_like(t1.flatten())
            removed_t2=np.zeros_like(t2.flatten())
            flattened_t1=t1.flatten()
            flattened_t2=t2.flatten()
            lenT1=len(removed_t1)
            for picky in picks:
                if picky<lenT1:
                    removed_t1[picky]=flattened_t1[picky]
                else:
                    removed_t2[picky-lenT1]=flattened_t2[picky-lenT1]
            removed_t1=removed_t1.reshape(t1.shape)
            removed_t2=removed_t2.reshape(t2.shape)
            t1_error = rhs_t.compute_t_1_amplitudes(f, system.u, t1, t2, system.o, system.v, np) #Original idea
            t2_error = rhs_t.compute_t_2_amplitudes(f, system.u, t1, t2, system.o, system.v, np) #Original idea
            #t1_error = rhs_t.compute_t_1_amplitudes(f, system.u, removed_t1, removed_t2, system.o, system.v, np) #Simens idea
            #t2_error = rhs_t.compute_t_2_amplitudes(f, system.u, removed_t1, removed_t2, system.o, system.v, np) #Simens idea
            ts=[np.concatenate((t1s[i],t2s[i]),axis=None) for i in range(len(t1s))]
            t_error=np.concatenate((t1_error,t2_error),axis=None)
            projection_errors=np.zeros(len(t1s))

            for i in range(len(projection_errors)):
                #projection_errors[i]=contract("ia,ia->",t1_error,t1s[i])+contract("ijab,ijab->",t2_error,t2s[i])
                projection_errors[i]=t_error[picks]@ts[i][picks]
            return projection_errors
        errors=erf(params)
        return errors
    def jacobian_function(params,system,t1s,t2s,t1_Jac,t2_Jac,picks):
        t1=np.zeros(t1s[0].shape)
        t2=np.zeros(t2s[0].shape)
        Ts=[]
        for i in range(len(t1s)):
            t1+=params[i]*t1s[i] #Starting guess
            t2+=params[i]*t2s[i] #Starting guess
            Ts.append(np.concatenate((t1s[i],t2s[i]),axis=None))
        jacobian_matrix=np.zeros((len(params),len(params)))
        t_Jac=np.concatenate((t1_Jac,t2_Jac),axis=None)
        for i in range(len(params)):
            for j in range(i,len(params)):
                #jacobian_matrix[j,i]=jacobian_matrix[i,j]=contract("ai,ai,ai->",t1s[i],t1s[j],t1_Jac)+contract("abij,abij,abij->",t2s[i],t2s[j],t2_Jac) #Use original Jacobian
                jacobian_matrix[j,i]=jacobian_matrix[i,j]=contract("i,i,i->",Ts[i][picks],Ts[j][picks],t_Jac[picks]) #Use jacobian based only on the used values
        return jacobian_matrix
    energy=[]
    start_guess=np.full(len(t1s),1/len(t1s))

    totamount=(len(t1s[0].flatten())+len(t2s[0].flatten()))
    random_number=totamount
    if random_picks>0:
        if random_picks>1:
            random_number=np.min((random_picks,totamount))
        else:
            random_number=int(random_picks*totamount)
    print("Picks: %f, Total amount: %f"%(random_number,totamount))
    picks=np.random.choice(totamount,random_number,replace=False)# Random choice
    #Alternatively, we cn take the 10 percent most important amplitudes!
    t1s,t2s=orthonormalize_ts(t1s,t2s)
    def pick_largest():
        T=np.zeros_like(np.concatenate((t1s[0],t2s[0]),axis=None))
        t1=np.zeros(t1s[0].shape)
        t2=np.zeros(t2s[0].shape)
        for i in range(len(t1s)):
            t1+=1/len(t1s)*t1s[i] #Starting guess
            t2+=1/len(t2s)*t2s[i] #Starting guess
            T+=np.concatenate((t1s[i],t2s[i]),axis=None)
        index_order=np.argsort(np.abs(T))
        return index_order[len(index_order)-random_number:], T[index_order[len(index_order)-random_number]]
    if optimal:
        picks,cutoff=pick_largest()
        print("Smallest picked: %e"%cutoff)
    for k,x_alpha in enumerate(x_alphas):
        if isinstance(reference_natorbs,list):
            ref_state=reference_natorbs[k]
        else:
            ref_state=reference_natorbs
        system = construct_pyscf_system_rhf_ref(
            molecule=molecule_func(*x_alpha),
            basis=basis,
            add_spin=False,
            anti_symmetrize=False,
            reference_state=ref_state,
            mix_states=mix_states,
            weights=weights,
            truncation=truncation
        )
        f = system.construct_fock_matrix(system.h, system.u)
        ESCF=system.compute_reference_energy().real
        t1_Jac,t2_Jac=system_jacobian(system)
        jacob=jacobian_function

        sol=root(error_function,start_guess,args=(system,t1s,t2s,t1_Jac,t2_Jac,picks),jac=jacob,method="hybr",options={"xtol":1e-2})#,method="broyden1")
        print("Error")
        print(error_function(sol.x,system,t1s,t2s,t1_Jac,t2_Jac,picks))
        try:
            print("Converged: ",sol.success, " number of iterations:",sol.nit)
        except:
            print("Converged: ",sol.success, " number of iterations:",sol.nfev)
        final=sol.x
        t1=np.zeros(t1s[0].shape)
        t2=np.zeros(t2s[0].shape)
        for i in range(len(t1s)):
            t1+=final[i]*t1s[i] #Starting guess
            t2+=final[i]*t2s[i] #Starting guess
        t1_error = rhs_t.compute_t_1_amplitudes(f, system.u, t1, t2, system.o, system.v, np) #Original idea
        t2_error = rhs_t.compute_t_2_amplitudes(f, system.u, t1, t2, system.o, system.v, np) #Original idea
        print("Maximal projection error:T1: %f, T2: %f"%(np.max(t1_error), np.max(t2_error)))
        newEn=rhs_e.compute_rccsd_ground_state_energy(f, system.u, t1, t2, system.o, system.v, np)+ESCF
        if sol.success==False:
            newEn=np.nan
        energy.append(newEn)
    return energy


def get_reference_determinant(molecule_func,refx,basis,charge=0):
    mol = gto.Mole()
    mol.unit = "bohr"
    mol.charge = charge
    mol.cart = False
    mol.build(atom=molecule_func(*refx), basis=basis)
    hf = scf.RHF(mol)
    hf.kernel()
    return np.asarray(hf.mo_coeff)
def get_natural_orbitals(molecule_func,xvals,basis,natorbs_ref,noons_ref,Sref):
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
        noons_ref,natorbs_ref=similiarize_natural_orbitals(noons_ref,natorbs_ref,noons,natorbs,mol.nelec,S,Sref)
        natorbs_list.append(natorbs_ref)
        Sref=S
        overlaps_list.append(S)
        noons_list.append(noons)
    return natorbs_list,noons_list
# System and basis parameters
if __name__=="__main__":
    basis = 'cc-pVDZ'
    basis_set = bse.get_basis(basis, fmt='nwchem')
    charge = 0
    #molecule =lambda arr: "Be 0.0 0.0 0.0; H 0.0 0.0 %f; H 0.0 0.0 -%f"%(arr,arr)
    molecule=lambda x:  "Be 0 0 0;H 0 0 %f;H 0 0 -%f"%(x,x)
    refx=[2]
    print(molecule(*refx))
    reference_determinant=get_reference_determinant(molecule,refx,basis,charge)
    sample_geom1=np.linspace(1.5,6,10)
    #sample_geom1=[2.5,3.0,6.0]
    sample_geom=[[x] for x in sample_geom1]
    sample_geom1=np.array(sample_geom).flatten()
    geom_alphas1=np.linspace(1.5,6.0,20)
    geom_alphas=[[x] for x in geom_alphas1]

    t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis_set,reference_determinant,mix_states=False,type="procrustes")
    energy_simen_random=solve_evc2(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=False,random_picks=0.1,optimal=True,type="procrustes",weights=weights)
    E_CCSDx,E_approx,E_diffguess,E_RHF,E_ownmethod=solve_evc(geom_alphas,molecule,basis_set,reference_determinant,t1s,t2s,l1s,l2s,mix_states=False,run_cc=True,cc_approx=False,type="procrustes")
    plt.plot(geom_alphas1,E_CCSDx,label="CCSD")
    plt.plot(geom_alphas1,E_approx,label="CCSD WF")
    plt.plot(geom_alphas1,energy_simen_random,label="CCSD AMP")
    plt.legend()
    plt.show()
