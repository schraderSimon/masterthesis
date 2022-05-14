import sys
sys.path.append("../libraries")
from quantum_library import *
import warnings
from scipy.io import loadmat, savemat
from scipy.stats import trimboth
from numpy.random import binomial
warnings.filterwarnings('ignore', category=DeprecationWarning)
class SamplerSystemGaussian():
    def filereader(self,infile):
        """Input: Infile
        Output:

        """
        sample_x=[]
        overlaps={}
        pauli_strings={}
        lines=infile.read()
        segments=lines.split("-\n") #Divide in segments of expectation valus
        information=segments[0]
        information_data=information.split("\n")[1:-1]
        for line in information_data:
            sample_x.append(float(line.split()[-1]))
        for segment in segments[1:-1]:
            segment_lines=segment.split("\n")[:-1]
            overlap=segment_lines[0].split(" ")
            i,j=int(overlap[1][0]),int(overlap[2][0])
            overlaps["%s%s"%(overlap[1][0],overlap[2][0])]=float(overlap[-1])
            dictian={}
            for line in segment_lines[2:]:
                pauli,coeff=line.split(" ")
                dictian[pauli]=float(coeff)
            pauli_strings["%d%d"%(i,j)]=dictian
        return sample_x,overlaps,pauli_strings

    def get_num_measurements(self):
        return sum(num_measurements_S.values())+sum(num_measurements_H.values())
    def remove_data(self,data_to_keep):
        """
        Which sample geometries to remove
        """
        data_to_keep=np.sort(data_to_keep).astype(int)
        self.sample_x=np.array(self.sample_x)[data_to_keep]
        new_ordering=np.arange(len(data_to_keep))
        overlaps_new={}
        pauli_string_exp_vals_new={}
        for ind_i,i in enumerate(data_to_keep):
            for ind_j,j in enumerate(data_to_keep):
                if j<i:
                    continue
                overlaps_new["%d%d"%(ind_i,ind_j)]=self.overlaps["%d%d"%(i,j)]
                pauli_string_exp_vals_new["%d%d"%(ind_i,ind_j)]=self.pauli_string_exp_vals["%d%d"%(i,j)]
        self.pauli_string_exp_vals=pauli_string_exp_vals_new
        self.overlaps=overlaps_new
    def __init__(self,filename,cutoff=None):
        infile=open(filename)
        self.sample_x,self.overlaps,self.pauli_string_exp_vals=self.filereader(infile)

    def create_measurement_data(self,num_measurements_S,num_measurements_H):
        """
        Initialize the S and H matrices with measurements.
        num_measurements_S[i,j] is how often we want to measure S[i,j], similarly for H.
        """
        sampled_pauli_string_exp_vals={}
        sampled_overlaps={}
        for key in self.overlaps: #For each combination i,j
            p=0.5*(1+self.overlaps[key]) #probability to measure 1
            new_p=binomial(num_measurements_S[key],p)/num_measurements_S[key] #Sampled probability using the binomial distribution
            sampled_overlaps[key]=2*new_p-1 #Obtain a new sampled overlap
        for key in self.pauli_string_exp_vals: #Again for "i,j"
            temp_dict={}
            for pauli_string in self.pauli_string_exp_vals[key]: #For each Pauli string
                p=0.5*(1+self.pauli_string_exp_vals[key][pauli_string]) #probability to measure 1
                new_p=binomial(num_measurements_H[key],p)/num_measurements_H[key]
                temp_dict[pauli_string]=2*new_p-1
            sampled_pauli_string_exp_vals[key]=temp_dict
        self.sampled_overlaps=sampled_overlaps
        self.sampled_pauli_string_exp_vals=sampled_pauli_string_exp_vals
        self.num_measurements_S=num_measurements_S
        self.num_measurements_H=num_measurements_H
    def add_measurements(self,type,index,number):
        """
        Adds "number" measurements for type "H" or "S" with index (i,j)
        """
        if type=="S": #If S should be measured
            p=0.5*(1+self.overlaps[index]) #True percentage
            sampled_p_new=binomial(number,p)/number #"number" new samples
            sampled_p_old=0.5*(1+self.sampled_overlaps[index])
            sampled_p_tot=(sampled_p_new*number+sampled_p_old*self.num_measurements_S[index])/(number+self.num_measurements_S[index])
            self.sampled_overlaps[index]=2*sampled_p_tot-1
            self.num_measurements_S[index]=self.num_measurements_S[index]+number
        elif type=="H": #If H should be measured
            for pauli_string in self.pauli_string_exp_vals[index]:
                p=0.5*(1+self.pauli_string_exp_vals[index][pauli_string]) #Percentage
                sampled_p_new=binomial(number,p)/number
                sampled_p_old=0.5*(1+self.sampled_pauli_string_exp_vals[index][pauli_string])
                sampled_p_tot=(sampled_p_new*number+sampled_p_old*self.num_measurements_H[index])/(number+self.num_measurements_H[index])
                self.sampled_pauli_string_exp_vals[index][pauli_string]=2*sampled_p_tot-1
            self.num_measurements_H[index]=self.num_measurements_H[index]+number
    def create_gaussian_dist(self,qubit_hamiltonian,nuc_rep):
        """
        Create gaussian distribution of data for mean and standard deviation for H, S
        """
        self.S_means={}
        self.H_means={}
        self.S_std={} # Not scaled
        self.H_std={} #Not scaled
        coefs_strings=qubit_hamiltonian.reduce().primitive.to_list()
        vals=list(map(list, zip(*coefs_strings)))[1]
        maxVar=np.sum(np.abs(np.array(vals))**2)
        maxSigma=np.sqrt(maxVar)
        for i in range(len(self.sample_x)):
            for j in range(i,len(self.sample_x)):
                val=self.sampled_overlaps["%d%d"%(i,j)]
                self.S_means["%d%d"%(i,j)]=val
                self.S_std["%d%d"%(i,j)]=1 #In reality MUCH lower

                self.S_std["%d%d"%(i,j)]=np.sqrt(0.5*(1-val)*(1-0.5*(1-val)))
                h=0
                #
                for pauliOpStr,coeff in coefs_strings:

                    h+=self.sampled_pauli_string_exp_vals["%d%d"%(i,j)][pauliOpStr]*coeff
                self.H_means["%d%d"%(i,j)]=np.real(h) #The number is real to begin with, but there are some 1j*1e-16
                self.H_std["%d%d"%(i,j)]=0
                H_variance=0
                for pauliOpStr,coeff in coefs_strings:
                    p=0.5*(1+self.sampled_pauli_string_exp_vals["%d%d"%(i,j)][pauliOpStr]) # Probability to measure
                    H_variance+=coeff**2*(p*(1-p))**2
                self.H_std["%d%d"%(i,j)]=np.sqrt(H_variance)
    def get_E(self,nuc_rep,threshold):
        """
        Returns the energy based on the sample mean
        """
        S=np.zeros((len(self.sample_x),len(self.sample_x)))
        H=np.zeros((len(self.sample_x),len(self.sample_x)))
        for i in range(len(self.sample_x)):
            for j in range(i,len(self.sample_x)):
                if i==j:
                    S[i,i]=1 #Should be 1 to begin with, jst to make sure
                else:
                    S[i,j]=S[j,i]=self.S_means["%d%d"%(i,j)]
                H[i,j]=H[j,i]=self.H_means["%d%d"%(i,j)]+S[j,i]*nuc_rep
        eigenvalue,vec=canonical_orthonormalization(H,S,threshold)
        return np.real(eigenvalue)
    def sample(self,nuc_rep,threshold):
        """
        Returns the energy based on the sample mean AND sample standard deviations, e.g. returns a random number. Threshold is the cutoff
        """
        S=np.zeros((len(self.sample_x),len(self.sample_x)))
        H=np.zeros((len(self.sample_x),len(self.sample_x)))
        for i in range(len(self.sample_x)):
            for j in range(i,len(self.sample_x)):
                if i==j:
                    S[i,i]=1
                    S_meanval=1
                    S_stdval=0
                else:
                    S_meanval=np.real(self.S_means["%d%d"%(i,j)])
                    S_stdval=np.real(self.S_std["%d%d"%(i,j)]/np.sqrt(self.num_measurements_S["%d%d"%(i,j)]))
                    S[i,j]=S[j,i]=np.random.normal(S_meanval,S_stdval)
                H_meanval=np.real(self.H_means["%d%d"%(i,j)]+S[j,i]*nuc_rep)
                H_stdval=np.real(self.H_std["%d%d"%(i,j)]/np.sqrt(self.num_measurements_H["%d%d"%(i,j)]))
                H[i,j]=H[j,i]=np.random.normal(H_meanval,np.sqrt(H_stdval**2+nuc_rep**2*S_stdval**2))
        eigenvalue,vec=canonical_orthonormalization(H,S,threshold)
        return np.real(eigenvalue)
    def exact_EVC(self,qubit_hamiltonian,nuc_rep,threshold):
        """
        Returns the exact EVC energy
        """
        S=np.zeros((len(self.sample_x),len(self.sample_x)))
        H=np.zeros((len(self.sample_x),len(self.sample_x)))
        coefs_strings=qubit_hamiltonian.reduce().primitive.to_list()
        for i in range(len(self.sample_x)):
            for j in range(i,len(self.sample_x)):
                S[i,j]=S[j,i]=self.overlaps["%d%d"%(i,j)]
                h=0
                for pauliOpStr,coeff in coefs_strings:
                    h+=self.pauli_string_exp_vals["%d%d"%(i,j)][pauliOpStr]*coeff
                H[i,j]=H[j,i]=np.real(h)+S[j,i]*nuc_rep
        print(H)
        print(S)
        eigenvalue,vec=canonical_orthonormalization(H,S,threshold)
        return np.real(eigenvalue)
    def grab_UCC2_val(self,qubit_hamiltonian,nuc_rep,k):
        """Return the UCC2 energy of the qubit Hamilonian, given the expectation values of the Pauli strings#"""
        h=nuc_rep
        coefs_strings=qubit_hamiltonian.reduce().primitive.to_list()
        for pauliOpStr,coeff in coefs_strings:
            h+=self.pauli_string_exp_vals["%d%d"%(k,k)][pauliOpStr]*coeff
        return np.real(h),np.real(numPyEigensolver.compute_eigenvalues(qubit_hamiltonian).eigenvalues[0])+nuc_rep
    def add_fake_measurements(self,type,index,number):
        """
        Increases the number of measurement for an index without actually changing data (adds "fake" measurements)
        """
        if type=="S":
            self.num_measurements_S[index]=self.num_measurements_S[index]+number
        elif type=="H":
            self.num_measurements_H[index]=self.num_measurements_H[index]+number
    def bootstrap(self,nuc_rep,threshold,num_bootstraps):
        eigvals=np.zeros(num_bootstraps)
        for i in range(num_bootstraps):
            eigvals[i]=self.sample(nuc_rep,threshold) #Get num_bootstraps estimates of eigenvalues
        E_mean=np.mean(eigvals)
        E_std=np.std(eigvals)
        return E_mean,E_std #Return mean and standard deviation of those eigenvalues
    def gradient_numMeas(self,eigval_threshold,nuc_rep,num_bootstraps,grad_increase_factor):
        """
        measures the "gradients", e.g. the estimated improvements when adding "grad_increase_factor" sapmles
        """
        S_gradients={}
        H_gradients={}
        mean,std=self.bootstrap(nuc_rep,threshold,num_bootstraps) #Get mean and standard deviation of eigenvalues.
        for i in range(0,len(self.sample_x)):
            for j in range(i+1,len(self.sample_x)): #Off-diagonal elements only, as diagonal elements are 1 by definition
                self.add_fake_measurements("S","%d%d"%(i,j),grad_increase_factor)
                mean_ij,std_ij=self.bootstrap(nuc_rep,threshold,num_bootstraps) #Estimate mean and standard deviation when adding eigenvalues based on Gaussian dist.
                S_gradients["%d%d"%(i,j)]=std_ij-std #Reduction in standard deviation
                self.add_fake_measurements("S","%d%d"%(i,j),-grad_increase_factor) #Remove estimates again
        for i in range(0,len(self.sample_x)):
            for j in range(i,len(self.sample_x)):
                self.add_fake_measurements("H","%d%d"%(i,j),grad_increase_factor)
                mean_ij,std_ij=self.bootstrap(nuc_rep,threshold,num_bootstraps)
                H_gradients["%d%d"%(i,j)]=std_ij-std
                self.add_fake_measurements("H","%d%d"%(i,j),-grad_increase_factor)
        return S_gradients,H_gradients
    def increase_max_gradient(self,nuc_rep,eigval_threshold,num_bootstraps,grad_increase_factor,increase_factor):
        """Estimate where adding more samples has the highest effect and add samples there"""
        S_gradients,H_gradients=self.gradient_numMeas(eigval_threshold,nuc_rep,num_bootstraps,grad_increase_factor)
        min_S = min(S_gradients, key=S_gradients.get)
        min_H = min(H_gradients, key=H_gradients.get)
        if S_gradients[min_S]<H_gradients[min_H]:
            self.add_measurements("S",min_S,increase_factor)
        else:
            #self.add_measurements("S",min_S,increase_factor)
            self.add_measurements("H",min_H,increase_factor)
if __name__=="__main__":
    def molecule(x):
        return "Be 0 0 0; H 0 0 %f; H 0 0 -%f"%(x,x)
    basis="STO-6G"
    ref_x=2
    mol = mol = gto.M(atom=molecule(ref_x), basis=basis,unit="Bohr")
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    nuc_rep=mf.energy_nuc()
    ref_det=mf.mo_coeff
    active_space=[1,2,3,4,5,6] #Freezing [1,2,3,5,6] works kinda for BeH2
    nelec=4
    x_of_interest=np.linspace(1.5,6.5,51)
    E_EVC=np.zeros(len(x_of_interest))
    E_exact=np.zeros(len(x_of_interest))
    E_UCC=np.zeros(len(x_of_interest))
    E_k2=np.zeros(len(x_of_interest))
    backend= AerSimulator(method='statevector',max_parallel_threads=4)
    seed=np.random.randint(100000)
    qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

    numPyEigensolver=NumPyEigensolver()

    #qubit_converter_nosymmetry = QubitConverter(mapper=JordanWignerMapper())
    qubit_converter_symmetry=tapering_value=[1,1,1]
    qubit_converter_symmetry=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True,z2symmetry_reduction=tapering_value)


    E_exact=np.zeros(len(x_of_interest))
    qubit_hamiltonians=[]
    nuc_reps=[]
    sample_qubit_hamiltonians=[]
    sample_nuc_reps=[]
    UCC2_energies=[]
    sample_exact=[]

    #Prepare data at points of interest (hamiltonians & nuclear repulsions)
    import pickle
    file="energy_data/BeH2_stretch_Hamiltonians_15-65.bin" #Contains parameters for BeH2 hamiltonian coefficients from 1.5 to 6.5
    try: #If data has already been created
        with open(file,"rb") as f:
            dictionary=pickle.load(f)
        nuc_reps=dictionary["nuclear_repulsions"]
        E_exact=dictionary["FCI"]
        qubit_hamiltonians=dictionary["hamiltonians"]
    except FileNotFoundError: #Create file
        for k,x in enumerate(x_of_interest):
            hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,x,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
            qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
            result = numPyEigensolver.compute_eigenvalues(qubit_hamiltonian)
            E_exact[k]=np.real(result.eigenvalues[0]+nuc_rep)
            qubit_hamiltonians.append(qubit_hamiltonian)
            nuc_reps.append(nuc_rep)
        dictionary={}
        dictionary["FCI"]=E_exact
        dictionary["nuclear_repulsions"]=nuc_reps
        dictionary["hamiltonians"]=qubit_hamiltonians

        with open(file,"wb") as f:
            pickle.dump(dictionary,f)
    #The File BeH2_UCC2_sampleXPauliStrings.txt contains UCC2 expectation values for all Pauli matrices in the hamiltonian
    samplersystem=SamplerSystemGaussian("data/BeH2_UCC2_sampleXPauliStrings.txt",cutoff=None)
    sample_x=samplersystem.sample_x

    #Get Hamiltonians and nuclear repulsions at sample geometries and UCC2 parameters
    for k,x in enumerate(sample_x):
        hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,x,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
        qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
        sample_qubit_hamiltonians.append(qubit_hamiltonian)
        sample_nuc_reps.append(nuc_rep)
        ucc2,exact=samplersystem.grab_UCC2_val(qubit_hamiltonian,nuc_rep,k)
        UCC2_energies.append(ucc2)
        sample_exact.append(exact)
    try:
        epsilon=int(sys.argv[1])
    except:
        epsilon=-2
    threshold=10**(epsilon) #Seems about reasonable :)
    samplersystem=SamplerSystemGaussian("data/BeH2_UCC2_sampleXPauliStrings.txt",cutoff=None) #refresh
    samplersystem.remove_data([0,1,2,3,5]) #Drop sample geometry x=2.5 which is included in BeH2_UCC2_sampleXPauliStrings.txt
    num_measurements_S={}
    num_measurements_H={}
    if epsilon==-5:
        num_start_measurements=4*1e6#5*1e5#Use for epsilon=1e-2
        measurement_increase=2*1e6 #1*1e5#Use for epsilon=1e-2
    if epsilon==-2:
        num_start_measurements=2*1e5#5*1e5#Use for epsilon=1e-2
        measurement_increase=1*1e5 #1*1e5#Use for epsilon=1e-2
    for i in range(len(samplersystem.sample_x)):
        for j in range(i,len(samplersystem.sample_x)):
            if i==j:
                num_measurements_S["%d%d"%(i,j)]=1 #Diagonal elements are always 1 for the overlap
            else:
                num_measurements_S["%d%d"%(i,j)]=int(num_start_measurements)
            num_measurements_H["%d%d"%(i,j)]=int(num_start_measurements)
    samplersystem.create_measurement_data(num_measurements_S,num_measurements_H) #Initialize the matrices S and H
    standard_dev=100
    Es=[]
    stds=[]
    num_measurements=[]
    chem_acc=1.6*1e-3
    multiplier=2
    noChange=0
    previous=100
    val=0
    while True:
        samplersystem.create_gaussian_dist(qubit_hamiltonians[val],nuc_reps[val]) #More or less randomly pick the last one, any would do
        samplersystem.increase_max_gradient(nuc_reps[val],threshold,500,int(measurement_increase),int(measurement_increase))
        E,std=samplersystem.bootstrap(nuc_reps[val],threshold,500)
        standard_dev=std
        if np.abs(previous-E)<chem_acc:
            noChange+=1
        else:
            noChange=0
            print("bæmp")
        previous=E
        Es.append(E)
        stds.append(std)
        print(E,std,samplersystem.H_std["%d%d"%(0,0)]/np.sqrt(samplersystem.num_measurements_H["%d%d"%(0,0)]))
        num_measurements.append(samplersystem.get_num_measurements())
        if noChange>20 and standard_dev<chem_acc/multiplier:
            break
    Es=np.array(Es)
    stds=np.array(stds)
    EVC_approx=[]
    EVC_std=[]
    print("bæmp")
    for k,x in enumerate(x_of_interest):
        print(x)
        samplersystem.create_gaussian_dist(qubit_hamiltonians[k],nuc_reps[k])
        E,std=samplersystem.bootstrap(nuc_reps[k],threshold,100)
        while std>chem_acc/multiplier:
            break
            samplersystem.increase_max_gradient(nuc_reps[k],threshold,100,int(measurement_increase),int(measurement_increase))

            E,std=samplersystem.bootstrap(nuc_reps[k],threshold,100)
            E=samplersystem.get_E(nuc_reps[k],threshold)
            print(E,std,samplersystem.H_std["%d%d"%(0,0)]/np.sqrt(samplersystem.num_measurements_H["%d%d"%(0,0)]))
            break
        E=samplersystem.get_E(nuc_reps[k],threshold)
        EVC_approx.append(E)
        EVC_std.append(std)
    print("Final number of measurements: %d"%samplersystem.get_num_measurements())
    data={}
    data["xvals"]=x_of_interest
    data["EVC_approx"]=EVC_approx
    data["EVC_approx_std"]=EVC_std
    data["num_measurements"]=num_measurements
    data["Es"]=Es
    data["stds"]=stds

    file="energy_data/BeH2_stretch_sample_%d.bin"%epsilon

    with open(file,"wb") as f:
        pickle.dump(data,f)
