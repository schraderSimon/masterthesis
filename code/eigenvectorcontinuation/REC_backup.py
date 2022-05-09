class eigvecsolver_RHF():
    def __init__(self,sample_points,basis_type,molecule=lambda x: "H 0 0 0 ; F 0 0 %d"%x,spin=0,unit='AU',charge=0,symmetry=False):
        """Initiate the solver.
        It Creates the HF coefficient matrices for the sample points for a given molecule.

        Input:
        sample_points (array) - the points at which to evaluate the WF
        basis_type - atomic basis
        molecule - A function f(x), returning the molecule as function of x
        """
        self.HF_coefficients=[] #The Hartree Fock coefficients solved at the sample points
        self.molecule=molecule
        self.basis_type=basis_type
        self.spin=spin
        self.unit=unit
        self.symmetry=symmetry
        self.charge=charge
        self.sample_points=sample_points
        self.solve_HF()
    def solve_HF(self):
        """
        Create coefficient matrices for each sample point
        """
        HF_coefficients=[]
        for x in self.sample_points:
            mol=self.build_molecule(x)
            mf=mol.RHF().run(verbose=2,conv_tol=1e-8)
            expansion_coefficients_mol= mf.mo_coeff[:, mf.mo_occ >=-1] #Use all
            HF_coefficients.append(expansion_coefficients_mol)
        self.HF_coefficients=HF_coefficients
        self.nMO_h=HF_coefficients[0].shape[0] #Number of molecular orbitals divided by two (for each spin)
    def build_molecule(self,x):
        """Create a molecule object with parameter x"""

        mol=gto.Mole()
        mol.atom="""%s"""%(self.molecule(x))
        mol.charge=self.charge
        mol.spin=self.spin
        mol.unit=self.unit
        mol.symmetry=self.symmetry
        mol.basis=self.basis_type
        mol.build()
        self.number_electronshalf=int(mol.nelectron/2)
        self.ne_h=self.number_electronshalf
        return mol
    def calculate_energies(self,xc_array):
        """Calculates the molecule's energy"""
        energy_array=np.zeros(len(xc_array))
        eigvec_array=[]
        for index,xc in enumerate(xc_array):
            mol_xc=self.build_molecule(xc)
            new_HF_coefficients=[]
            print("Pre basis change")
            print(self.HF_coefficients[0])
            for i in range(len(self.sample_points)):
                new_HF_coefficients.append(self.basischange(self.HF_coefficients[i],mol_xc.intor("int1e_ovlp")))[:,:self.number_electronshalf]
            print("Post basis change")
            print(new_HF_coefficients[0])
            S,T=self.calculate_ST_matrices(mol_xc,new_HF_coefficients)
            try:
                eigval,eigvec=generalized_eigenvector(T,S)
            except:
                eigval=float('NaN')
                eigvec=float('NaN')
            energy_array[index]=eigval
            eigvec_array.append(eigvec)
        return energy_array,eigvec_array

    def getdeterminant_matrix(self,AO_overlap,HF_coefficients_left,HF_coefficients_right):
        determinant_matrix=np.einsum("ab,ai,bj->ij",AO_overlap,HF_coefficients_left,HF_coefficients_right)
        return determinant_matrix

    def biorthogonalize(self,C_w,C_x,AO_overlap):
        """Biorthogonalizes the problem
        Input: The left and right Slater determinants (in forms of coefficient matrixes)
        Returns: The diagonal matrix (as array) D, the new basises for C_w and C_x.
        """
        S=self.getdeterminant_matrix(AO_overlap,C_w,C_x);
        Linv,D,Uinv=LDU_decomp(S)
        C_w=C_w@Linv
        C_x=C_x@Uinv.T
        return D,C_w,C_x
    def getoverlap(self,determinant_matrix):
        overlap=np.linalg.det(determinant_matrix)**2
        return overlap
    def calculate_overlap_to_HF(self,xc_array):
        energies,eigvecs=self.calculate_energies(xc_array) #get energies and eigenvalues
        overlap_to_HF=np.zeros(len(energies))
        for index,xc in enumerate(xc_array):
            mol_xc=self.build_molecule(xc)

            "Calculate RHF solution"
            mol_xc=self.build_molecule(xc)
            overlap_basis=mol_xc.intor("int1e_ovlp")
            mf=mol_xc.RHF().run(verbose=2,conv_tol=1e-8)
            true_HF_coeffmatrix= mf.mo_coeff[:, mf.mo_occ >=0.5] #Use all of them :)
            "re-calculate HF coefficients"
            overlaps=np.zeros(len(self.sample_points))
            for i in range(len(self.sample_points)):
                conv_sol=self.basischange(self.HF_coefficients[i][:,:self.number_electronshalf],mol_xc.intor("int1e_ovlp"))
                determinant_matrix=self.getdeterminant_matrix(overlap_basis,true_HF_coeffmatrix,conv_sol)
                overlap=self.getoverlap(determinant_matrix)
                overlaps[i]=overlap
            overlap_to_HF[index]=np.dot(overlaps,eigvecs[index])
        return overlap_to_HF
    def calculate_ST_matrices(self,mol_xc,new_HF_coefficients):
        number_electronshalf=self.number_electronshalf
        overlap_basis=mol_xc.intor("int1e_ovlp")
        energy_basis_1e=mol_xc.intor("int1e_kin")+mol_xc.intor("int1e_nuc")
        energy_basis_2e=mol_xc.intor('int2e',aosym="s1")
        Smat=np.zeros((len(self.HF_coefficients),len(self.HF_coefficients)))
        Tmat=np.zeros((len(self.HF_coefficients),len(self.HF_coefficients)))
        for i in range(len(self.HF_coefficients)):
            for j in range(i,len(self.HF_coefficients)):
                D,C_w,C_x=self.biorthogonalize(new_HF_coefficients[i],new_HF_coefficients[j],overlap_basis)
                D_tilde=D[np.abs(D)>1e-15]
                S_tilde=np.prod(D_tilde)**2 #Squared because this is RHF.
                S=np.prod(D)**2
                Smat[i,j]=Smat[j,i]=S
                #energy_Hurensohn=self.energy_func_naive(C_w,C_x,energy_basis_1e,energy_basis_2e,D)*S_tilde
                energy=self.energy_func(C_w,C_x,energy_basis_1e,energy_basis_2e,D)*S_tilde
                nuc_repulsion_energy=mol_xc.energy_nuc()*S
                Tmat[i,j]=Tmat[j,i]=energy+nuc_repulsion_energy
        return Smat,Tmat
    def energy_func_naive(self,C_w,C_x,onebodyp,twobodyp,D):
        twobody=ao2mo.get_mo_eri(twobodyp,(C_w,C_w,C_x,C_x),aosym="s1")
        onebody=np.einsum("ki,lj,kl->ij",C_w,C_x,onebodyp)
        alpha=np.arange(5)
        beta=np.arange(5)
        onebody_alpha=np.sum(np.diag(onebody)[alpha])
        onebody_beta=np.sum(np.diag(onebody)[beta])
        energy2=0
        #Equation 2.175 from Szabo, Ostlund
        for a in alpha:
            for b in alpha:
                energy2+=twobody[a,a,b,b]
                energy2-=twobody[a,b,b,a]
        for a in alpha:
            for b in beta:
                energy2+=twobody[a,a,b,b]
        for a in beta:
            for b in alpha:
                energy2+=twobody[a,a,b,b]
        for a in beta:
            for b in beta:
                energy2+=twobody[a,a,b,b]
                energy2-=twobody[a,b,b,a]
        energy2*=0.5
        return onebody_alpha+onebody_beta+energy2
    def energy_func(self,C_w,C_x,onebody,twobody,D):
        number_of_zeros=len(D[np.abs(D)<1e-15])*2
        zero_indices=np.where(np.abs(D)<1e-15)
        if number_of_zeros>2:
            return 0
        else:
            P_s=[]
            for i in range(self.number_electronshalf):
                P_s.append(np.einsum("m,v->mv",C_w[:,i],C_x[:,i]))
            P=P_s
            for i in range(self.number_electronshalf):
                P[i]=P[i]/D[i]
            W=np.sum(P,axis=0).T
            if number_of_zeros==2:
                print("Bæmp")
                H1=0
                i=zero_indices[0][0]
                J_contr=np.einsum("ms,ms->",P_s[i],np.einsum("vt,msvt->ms",P_s[i],twobody))
                K_contr=np.einsum("ms,ms->",P_s[i],np.einsum("vt,mtvs->ms",P_s[i],twobody))
                H2=J_contr-K_contr
            elif number_of_zeros==0:
                J_contr=4*np.einsum("sm,ms->",W,np.einsum("vt,msvt->ms",W,twobody))
                K_contr=2*np.einsum("sm,ms->",W,np.einsum("vt,mtvs->ms",W,twobody))
                H2=0.5*(J_contr-K_contr)
                H1=2*np.einsum("mv,mv->",W,onebody)
            return H2+H1
    def basischange(self,C_old,overlap_AOs_newnew):
        S=np.einsum("mi,vj,mv->ij",C_old,C_old,overlap_AOs_newnew)
        S_eig,S_U=np.linalg.eigh(S)
        S_powerminusonehalf=S_U@np.diag(S_eig**(-0.5))@S_U.T
        C_new=np.einsum("ij,mj->mi",S_powerminusonehalf,C_old)
        return C_new
