from measure_overlap import *
from pyscf import gto, scf, ao2mo, fci, mcscf
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, cc,scf, ao2mo,fci
np.set_printoptions(linewidth=300,precision=10,suppress=True)
from scipy.linalg import block_diag, eig, orth
from numba import jit
from matrix_operations import *
from helper_functions import *
from scipy.optimize import minimize, root,newton
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy, ParticleNumber
from qiskit_nature.properties.second_quantization.electronic.electronic_structure_driver_result import ElectronicStructureDriverResult
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis, ElectronicBasisTransform

from qiskit_nature.properties import GroupedProperty
import numpy as np
import matplotlib.pyplot as plt
from qiskit.algorithms import NumPyEigensolver,VQE
#from qiskit.aqua.algorithms import VQE
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    ElectronicIntegrals,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
    IntegralProperty,
)
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit import Aer, transpile
from qiskit.providers.aer import AerSimulator

from qiskit.algorithms.optimizers import *
from qiskit_nature.circuit.library import UCC,UCCSD, HartreeFock, PUCCD
from qiskit.circuit.library import EfficientSU2, QAOAAnsatz
from qiskit_nature.algorithms import VQEUCCFactory
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper,BravyiKitaevMapper, BravyiKitaevSuperFastMapper
from qiskit.utils import QuantumInstance
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.opflow import X, Y, Z, I
from qiskit.opflow import CircuitStateFn, StateFn, TensoredOp
from qiskit.opflow import PauliExpectation, CircuitSampler, StateFn, AerPauliExpectation, MatrixExpectation
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.converters import circuit_to_gate
from qiskit.tools.visualization import circuit_drawer
from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer, FreezeCoreTransformer
from qiskit.opflow.primitive_ops import Z2Symmetries, PauliOp
import sys
import warnings
from scipy.io import loadmat, savemat
from scipy.stats import trimboth
from numpy.random import binomial
warnings.filterwarnings('ignore', category=DeprecationWarning)
def generalized_eigenvector(T,S,threshold=1e-8,symmetric=True):
    """Solves the generalized eigenvector problem.
    Input:
    T: The symmetric matrix
    S: The overlap matrix
    symmetric: Wether the matrices T, S are symmetric
    Returns: The lowest eigenvalue & eigenvector
    """
    ###The purpose of this procedure here is to remove all very small eigenvalues of the overlap matrix for stability
    s, U=np.linalg.eigh(S) #Diagonalize S (overlap matrix, Hermitian by definition)
    U=np.fliplr(U)
    s=s[::-1] #Order from largest to lowest; S is an overlap matrix, hence we won't
    s=s[s>threshold] #Keep only largest eigenvalues
    spowerminushalf=s**(-0.5) #Take s
    snew=np.zeros((len(U),len(spowerminushalf)))
    sold=np.diag(spowerminushalf)
    snew[:len(s),:]=sold
    s=snew


    ###Canonical orthogonalization
    X=U@s
    Tstrek=X.T@T@X
    if symmetric:
        epsilon, Cstrek = np.linalg.eigh(Tstrek)
    else:
        epsilon, Cstrek = np.linalg.eig(Tstrek)
    idx = epsilon.argsort()[::1] #Order by size (non-absolute)
    epsilon = epsilon[idx]
    Cstrek = Cstrek[:,idx]
    C=X@Cstrek
    lowest_eigenvalue=epsilon[0]
    lowest_eigenvector=C[:,0]
    return lowest_eigenvalue,lowest_eigenvector

def filereader(infile):
    sample_x=[]
    overlaps={}
    pauli_strings={}
    lines=infile.read()
    segments=lines.split("-\n")
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
def create_characteristic_data(sample_x,overlaps,pauli_string_exp_vals,num_measurements):
    sampled_pauli_string_exp_vals={}
    sampled_overlaps={}
    for key in overlaps:
        p=0.5*(1-overlaps[key]) #Percentage
        new_p=binomial(num_measurements,p)/num_measurements
        sampled_overlaps[key]=2*new_p-1
    for key in pauli_string_exp_vals:
        temp_dict={}
        for pauli_string in pauli_string_exp_vals[key]:
            p=0.5*(1-pauli_string_exp_vals[key][pauli_string]) #Percentage
            new_p=binomial(num_measurements,p)/num_measurements
            temp_dict[pauli_string]=2*new_p-1
        sampled_pauli_string_exp_vals[key]=temp_dict
    return sampled_overlaps,sampled_pauli_string_exp_vals
def create_measurement_data(sample_x,overlaps,pauli_string_exp_vals,num_measurements_S,num_measurements_H):
    sampled_pauli_string_exp_vals={}
    sampled_overlaps={}
    for key in overlaps:
        p=0.5*(1-overlaps[key]) #Percentage
        new_p=binomial(num_measurements_S[key],p)/num_measurements_S[key]
        sampled_overlaps[key]=2*new_p-1
    for key in pauli_string_exp_vals:
        temp_dict={}
        for pauli_string in pauli_string_exp_vals[key]:
            p=0.5*(1-pauli_string_exp_vals[key][pauli_string]) #Percentage
            new_p=binomial(num_measurements_H[key],p)/num_measurements_H[key]
            temp_dict[pauli_string]=2*new_p-1
        sampled_pauli_string_exp_vals[key]=temp_dict
    return sampled_overlaps,sampled_pauli_string_exp_vals
def add_measurements(sample_x,overlaps,pauli_string_exp_vals,sampled_overlaps,sampled_pauli_string_exp_vals,num_measurements_S,num_measurements_H,type,index,number):
    pass
class SamplerSystemGaussian():
    def get_num_measurements(self):
        return sum(num_measurements_S.values())+sum(num_measurements_H.values())
    def remove_data(self,data_to_keep):
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
    def remove_least_important(self,threshold):
        Smat=np.zeros((len(self.sample_x),len(self.sample_x)))
        for i in range(len(self.sample_x)):
            for j in range(i,len(self.sample_x)):
                Smat[i,j]=Smat[j,i]=self.sampled_overlaps["%d%d"%(i,j)]
        q,r=np.linalg.qr(Smat)
        removerinos=np.where(abs(np.diag(r))<threshold)
        data_to_keep=np.delete(np.arange(len(self.sample_x)),removerinos)
        print(r)
        self.remove_data(np.array(data_to_keep).astype(int))
    def __init__(self,filename,cutoff=None):
        infile=open(filename)
        self.sample_x,self.overlaps,self.pauli_string_exp_vals=filereader(infile)

    def create_measurement_data(self,num_measurements_S,num_measurements_H):
        sampled_pauli_string_exp_vals={}
        sampled_overlaps={}
        for key in self.overlaps:
            p=0.5*(1+self.overlaps[key]) #Percentage
            new_p=binomial(num_measurements_S[key],p)/num_measurements_S[key]
            sampled_overlaps[key]=2*new_p-1
        for key in self.pauli_string_exp_vals:
            temp_dict={}
            for pauli_string in self.pauli_string_exp_vals[key]:
                p=0.5*(1+self.pauli_string_exp_vals[key][pauli_string]) #Percentage
                new_p=binomial(num_measurements_H[key],p)/num_measurements_H[key]
                temp_dict[pauli_string]=2*new_p-1
            sampled_pauli_string_exp_vals[key]=temp_dict
        self.sampled_overlaps=sampled_overlaps
        self.sampled_pauli_string_exp_vals=sampled_pauli_string_exp_vals
        self.num_measurements_S=num_measurements_S
        self.num_measurements_H=num_measurements_H
    def add_measurements(self,type,index,number):
        if type=="S":
            p=0.5*(1+self.overlaps[index]) #Percentage
            sampled_p_new=binomial(number,p)/number #"number" new samples
            sampled_p_old=0.5*(1+self.sampled_overlaps[index])
            sampled_p_tot=(sampled_p_new*number+sampled_p_old*self.num_measurements_S[index])/(number+self.num_measurements_S[index])
            self.sampled_overlaps[index]=2*sampled_p_tot-1
            self.num_measurements_S[index]=self.num_measurements_S[index]+number
        elif type=="H":
            for pauli_string in self.pauli_string_exp_vals[index]:
                p=0.5*(1+self.pauli_string_exp_vals[index][pauli_string]) #Percentage
                sampled_p_new=binomial(number,p)/number
                sampled_p_old=0.5*(1+self.sampled_pauli_string_exp_vals[index][pauli_string])
                sampled_p_tot=(sampled_p_new*number+sampled_p_old*self.num_measurements_H[index])/(number+self.num_measurements_H[index])
                self.sampled_pauli_string_exp_vals[index][pauli_string]=2*sampled_p_tot-1
            self.num_measurements_H[index]=self.num_measurements_H[index]+number
        #print(self.num_measurements_H)
        #print(self.num_measurements_S)
    def create_gaussian_dist(self,qubit_hamiltonian,nuc_rep):
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
                self.S_means["%d%d"%(i,j)]=self.sampled_overlaps["%d%d"%(i,j)]
                self.S_std["%d%d"%(i,j)]=1 #In reality MUCH lower
                self.S_std["%d%d"%(i,j)]=np.sqrt(0.5*(1-self.sampled_overlaps["%d%d"%(i,j)])*(1-0.5*(1-self.sampled_overlaps["%d%d"%(i,j)])))
                h=0
                self.H_std["%d%d"%(i,j)]=0
                for pauliOpStr,coeff in coefs_strings:
                    h+=self.pauli_string_exp_vals["%d%d"%(i,j)][pauliOpStr]*coeff
                self.H_means["%d%d"%(i,j)]=np.real(h)
                self.H_std["%d%d"%(i,j)]=maxSigma #In reality MUCH lower
    def sample(self,nuc_rep,threshold):
        S=np.zeros((len(self.sample_x),len(self.sample_x)))
        H=np.zeros((len(self.sample_x),len(self.sample_x)))
        for i in range(len(self.sample_x)):
            for j in range(i,len(self.sample_x)):
                if i==j:
                    S[i,i]=1
                else:
                    S[i,j]=S[j,i]=np.random.normal(self.S_means["%d%d"%(i,j)],self.S_std["%d%d"%(i,j)]/np.sqrt(self.num_measurements_S["%d%d"%(i,j)]))
                H[i,j]=H[j,i]=np.random.normal(self.H_means["%d%d"%(i,j)]+S[j,i]*nuc_rep,self.H_std["%d%d"%(i,j)]/np.sqrt(self.num_measurements_H["%d%d"%(i,j)]))
        eigenvalue,vec=generalized_eigenvector(H,S,threshold)
        return np.real(eigenvalue)
    def exact_EVC(self,qubit_hamiltonian,nuc_rep,threshold):
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
        eigenvalue,vec=generalized_eigenvector(H,S,threshold)
        return np.real(eigenvalue)
    def grab_UCC2_val(self,qubit_hamiltonian,nuc_rep,k):
        h=nuc_rep
        coefs_strings=qubit_hamiltonian.reduce().primitive.to_list()
        for pauliOpStr,coeff in coefs_strings:
            h+=self.pauli_string_exp_vals["%d%d"%(k,k)][pauliOpStr]*coeff
        return np.real(h),np.real(numPyEigensolver.compute_eigenvalues(qubit_hamiltonian).eigenvalues[0])+nuc_rep
    def add_fake_measurements(self,type,index,number):
        if type=="S":
            self.num_measurements_S[index]=self.num_measurements_S[index]+number
        elif type=="H":
            self.num_measurements_H[index]=self.num_measurements_H[index]+number
    def bootstrap(self,nuc_rep,threshold,num_bootstraps):
        eigvals=np.zeros(num_bootstraps)
        for i in range(num_bootstraps):
            eigvals[i]=self.sample(nuc_rep,threshold)
        E_mean=np.mean(eigvals)
        E_std=np.std(eigvals)
        return E_mean,E_std
    def gradient_numMeas(self,eigval_threshold,nuc_rep,num_bootstraps,grad_increase_factor):
        S_gradients={}
        H_gradients={}
        mean,std=self.bootstrap(nuc_rep,threshold,num_bootstraps)
        for i in range(0,len(self.sample_x)):
            for j in range(i+1,len(self.sample_x)): #Off-diagonal elements only, as diagonal elements are 1 by definition
                self.add_fake_measurements("S","%d%d"%(i,j),grad_increase_factor)
                mean_ij,std_ij=self.bootstrap(nuc_rep,threshold,num_bootstraps)
                S_gradients["%d%d"%(i,j)]=std_ij-std
                self.add_fake_measurements("S","%d%d"%(i,j),-grad_increase_factor)
        for i in range(0,len(self.sample_x)):
            for j in range(i,len(self.sample_x)):
                self.add_fake_measurements("H","%d%d"%(i,j),grad_increase_factor)
                mean_ij,std_ij=self.bootstrap(nuc_rep,threshold,num_bootstraps)
                H_gradients["%d%d"%(i,j)]=std_ij-std
                self.add_fake_measurements("H","%d%d"%(i,j),-grad_increase_factor)
        return S_gradients,H_gradients
    def increase_max_gradient(self,nuc_rep,eigval_threshold,num_bootstraps,grad_increase_factor,increase_factor):
        S_gradients,H_gradients=self.gradient_numMeas(eigval_threshold,nuc_rep,num_bootstraps,grad_increase_factor)
        min_S = min(S_gradients, key=S_gradients.get)
        min_H = min(H_gradients, key=H_gradients.get)
        if S_gradients[min_S]<H_gradients[min_H]:
            self.add_measurements("S",min_S,increase_factor)
        else:
            self.add_measurements("H",min_H,increase_factor)
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

qubit_converter_nosymmetry = QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True)
tapering_value=[1,1,1]
qubit_converter_symmetry=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True,z2symmetry_reduction=tapering_value)

E_exact=np.zeros(len(x_of_interest))
qubit_hamiltonians=[]
nuc_reps=[]
sample_qubit_hamiltonians=[]
sample_nuc_reps=[]
UCC2_energies=[]
sample_exact=[]
for k,x in enumerate(x_of_interest):
    hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,x,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
    qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
    result = numPyEigensolver.compute_eigenvalues(qubit_hamiltonian)
    E_exact[k]=np.real(result.eigenvalues[0]+nuc_rep)
    qubit_hamiltonians.append(qubit_hamiltonian)
    nuc_reps.append(nuc_rep)
samplersystem=SamplerSystemGaussian("BeH2_UCC2_sampleXPauliStrings.txt",cutoff=None)
sample_x=samplersystem.sample_x
for k,x in enumerate(sample_x):
    hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,x,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
    qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
    sample_qubit_hamiltonians.append(qubit_hamiltonian)
    sample_nuc_reps.append(nuc_rep)
    ucc2,exact=samplersystem.grab_UCC2_val(qubit_hamiltonian,nuc_rep,k)
    UCC2_energies.append(ucc2)
    sample_exact.append(exact)
def plot_exact_energy_profile():

    eigenvalues_EVC_exact=np.zeros(len(x_of_interest))
    eigenvalues_EVC_oneRemoved=np.zeros(len(x_of_interest))
    for k,x in enumerate(x_of_interest):
        eigenvalues_EVC_exact[k]=samplersystem.exact_EVC(qubit_hamiltonians[k],nuc_reps[k],1e-6)
    samplersystem=SamplerSystemGaussian("BeH2_UCC2_sampleXPauliStrings.txt",cutoff=None)
    samplersystem.remove_data([2,3,4,5])
    for k,x in enumerate(x_of_interest):
        eigenvalues_EVC_oneRemoved[k]=samplersystem.exact_EVC(qubit_hamiltonians[k],nuc_reps[k],1e-6)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x_of_interest,E_exact,label="FCI",color="b")
    ax1.plot(x_of_interest,eigenvalues_EVC_exact,label="EVC (6)",color="r")
    ax1.plot(x_of_interest,eigenvalues_EVC_oneRemoved,label="EVC (4)",color="g")
    ax1.plot(sample_x,UCC2_energies,"*",label="Sample points",color="m")
    ax1.set_title("Potential energy surface")
    ax1.set_ylabel("Energy (Hartree)")
    ax1.set_xlabel("Interatomic distance (Bohr)")
    ax1.legend()
    ax2.plot(x_of_interest,abs(np.array(eigenvalues_EVC_exact)-np.array(E_exact)),label="EVC (6)",color="r")
    ax2.plot(x_of_interest,abs(np.array(eigenvalues_EVC_oneRemoved)-np.array(E_exact)),label="EVC (4)",color="g")
    ax2.plot(sample_x,abs(np.array(UCC2_energies)-np.array(sample_exact)),"*",label="Sample points",color="m")
    ax2.set_title(r"Deviation from $E_{FCI}$")
    ax2.set_ylabel("Energy (Hartree)")
    ax2.set_xlabel("Interatomic distance (Bohr)")
    ax2.set_yscale('log')
    ax2.legend()
    plt.tight_layout()
    plt.savefig("Exact_EVC_BeH2.pdf")
    plt.show()
    return
#plot_exact_energy_profile()
print(E_exact)
threshold=10**(-1.5) #Seems about reasonable :)
samplersystem=SamplerSystemGaussian("BeH2_UCC2_sampleXPauliStrings.txt",cutoff=None)
num_measurements_S={}
num_measurements_H={}
for i in range(len(samplersystem.sample_x)):
    for j in range(i,len(samplersystem.sample_x)):
        if i==j:
            num_measurements_S["%d%d"%(i,j)]=1
        else:
            num_measurements_S["%d%d"%(i,j)]=int(1e5)
        num_measurements_H["%d%d"%(i,j)]=int(1e5)
samplersystem.create_measurement_data(num_measurements_S,num_measurements_H)
#samplersystem.remove_least_important(0.01)
#samplersystem.remove_data([3,4,5])
standard_dev=100
Es=[]
stds=[]
num_measurements=[]
chem_acc=1.6*1e-3
noChange=0
previous=100

while True:
    samplersystem.create_gaussian_dist(qubit_hamiltonians[-1],nuc_reps[-1])
    samplersystem.increase_max_gradient(nuc_reps[-1],threshold,100,int(1e6),int(1e6))
    E,std=samplersystem.bootstrap(nuc_reps[-1],threshold,100)
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
    if noChange>20 and standard_dev<chem_acc/3:
        break
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,8))
ax1.set_title("Convergence wrt. measurements")
ax1.axhline(E_exact[-1],label=r"$E_{FCI}$",color="b")
ax1.axhline(E_exact[-1]-chem_acc,label=r"Chemical accuracy",color="g")
ax1.axhline(E_exact[-1]+chem_acc,color="g")
ax1.set_ylim([-15.500,-15.490])
ax1.set_yticks(np.arange(-15.500, -15.490, step=0.001))
from matplotlib import ticker
formatter = ticker.FormatStrFormatter('%.4f')
ax1.yaxis.set_major_formatter(formatter)
ax1.plot(num_measurements,Es,label=r"$\bar E_0$",color="r")
ax1.fill_between(num_measurements, Es - 2*std, Es + 2*std,
                 color='r', alpha=0.2,label=r"$\pm2\sigma_{E_0}$")
ax1.set_xlabel(r"$N_{measurements}$")
ax1.set_ylabel("Energy (Hartree)")
ax1.legend()
EVC_approx=[]
EVC_std=[]
for k,x in enumerate(x_of_interest):
    samplersystem.create_gaussian_dist(qubit_hamiltonians[k],nuc_reps[k])
    E,std=samplersystem.bootstrap(nuc_reps[k],threshold,100)
    while std>chem_acc/3:
        samplersystem.increase_max_gradient(nuc_reps[k],threshold,100,int(1e6),int(1e6))
        E,std=samplersystem.bootstrap(nuc_reps[k],threshold,100)
    EVC_approx.append(E)
    EVC_std.append(std)
ax2.set_title(r"Deviation from $E_{FCI}$ along PES")

ax2.plot(x_of_interest,np.array(EVC_approx)-np.array(E_exact),label=r"$\bar{E_0}-E_{FCI}$")
ax2.fill_between(x_of_interest,np.array(EVC_approx)-np.array(E_exact)-2*np.array(EVC_std),np.array(EVC_approx)-np.array(E_exact)+2*np.array(EVC_std),color='r', alpha=0.2,label=r"$\pm2\sigma_{E_0}$")
ax2.set_xlabel("Interatomic distance")
ax2.set_ylabel("Energy (Hartree)")
ax2.legend()
plt.tight_layout()
plt.savefig("Approx_EVC_BeH2%.1e.pdf"%(threshold))
print("Final number of measurements: %d"%samplersystem.get_num_measurements())
plt.show()
