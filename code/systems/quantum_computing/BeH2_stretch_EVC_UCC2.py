"""
Based on UCC2 sampling data, find EVC values for linear stretch of BeH2.
"""
import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from quantum_library import *
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
try:
    sample_x=eval(sys.argv[1])
except:
    sample_x=[1.5,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,3.5,4.5,5.5,6.5]
x_of_interest=np.linspace(1.5,6.5,51)
E_EVC=np.zeros(len(x_of_interest))
E_exact=np.zeros(len(x_of_interest))
E_UCC=np.zeros(len(x_of_interest))
backend= Aer.get_backend('statevector_simulator')
seed=np.random.randint(100000)
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

optimizer=SciPyOptimizer("BFGS")#L_BFGS_B()
numPyEigensolver=NumPyEigensolver()
UCC_UCC2_params=loadmat("data/BeH2_UCC_vals.mat")
UCC_2_params=UCC_UCC2_params["UCC_2"]
UCC_1_params=UCC_UCC2_params["UCC_1"]
print(UCC_UCC2_params.keys())
xvals=UCC_UCC2_params["xvals"][0]


qubit_converter_nosymmetry = QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True)
tapering_value=[1,1,1]
qubit_converter_symmetry=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True,z2symmetry_reduction=tapering_value)

UCC_circuits=[]

#All of this just to get the number of qubits...
hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,3,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
qubit_hamiltonian=qubit_hamiltonian.reduce()
num_qubits=qubit_hamiltonian.num_qubits
#Get the circuits
qasm_strings=[]
for sample_x_val in sample_x: #Get circuits
    print("Baemp")
    idx = (np.abs(xvals - sample_x_val)).argmin() #Find the correct index
    x=xvals[idx]
    print(x)
    UCC_param=UCC_2_params[idx]
    print(num_particles,num_spin_orbitals,num_qubits,)
    UCC_ansatz_f,initial_point=UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=2,generalized=False)
    print(len(initial_point))
    print(len(UCC_param))
    UCC_circuits.append(UCC_ansatz_f.assign_parameters(UCC_param)) #Create UCC circuit and add to list
print("Done getting ciruicts")
for i in range(len(UCC_circuits)):
    newcirc=transpile(UCC_circuits[i],backend=backend,optimization_level=1) #Effectivize circuit
    UCC_circuits[i]=newcirc
    stringy=newcirc.qasm(formatted=False)
    qasm_strings.append(stringy)
dicterino={}
dicterino["xvals"]=sample_x
dicterino["qasm_strings"]=qasm_strings
savemat("data/BeH2_stretch_UCC2_qasms.mat",dicterino)
print("Done transforming circuits")
zero1states={}
pauli_string_exp_vals={}
overlaps={}
for i in range(len(sample_x)):
    for j in range(i,len(sample_x)):
        zero1states["%d%d"%(i,j)]=get_01_state(UCC_circuits[i],UCC_circuits[j],num_qubits,backend)
print("Done getting new states")
S=np.zeros((len(sample_x),len(sample_x)))
for i in range(len(sample_x)):
    for j in range(i,len(sample_x)):
        S[i,j]=S[j,i]=get_overlap_expectations(zero1states["%d%d"%(i,j)],num_qubits,qi)
print(S)
print(np.linalg.eigh(S)[0])
E_EVC=np.zeros(len(x_of_interest))
E_exact=np.zeros(len(x_of_interest))
UCC2_energies=[]
sample_exact=[]
for k,x in enumerate(sample_x):
    hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,x,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
    qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
    h=get_energy_expectations_01state(zero1states["%d%d"%(k,k)],num_qubits,qubit_hamiltonian,qi)
    UCC2_energies.append(h+nuc_rep)
    result = numPyEigensolver.compute_eigenvalues(qubit_hamiltonian)
    sample_exact.append(np.real(result.eigenvalues[0]+nuc_rep))
Hs=[]
Ss=[]
for k,x in enumerate(x_of_interest):
    print(k)
    H=np.zeros((len(sample_x),len(sample_x)))
    hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,x,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
    qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
    for i in range(len(sample_x)):
        for j in range(i,len(sample_x)):
            s=S[i,j]
            h=get_energy_expectations_01state(zero1states["%d%d"%(i,j)],num_qubits,qubit_hamiltonian,qi)
            H[i,j]=H[j,i]=np.real(h+S[i,j]*nuc_rep)
    Hs.append(H)
    Ss.append(S)
    print(H)
    e,c=generalized_eigenvector(H,S,threshold=1e-14)
    E_EVC[k]=np.real(e)
    result = numPyEigensolver.compute_eigenvalues(qubit_hamiltonian)
    E_exact[k]=np.real(result.eigenvalues[0]+nuc_rep)
    print(E_EVC[k],E_exact[k])
fig, (ax1, ax2) = plt.subplots(1, 2)
import pickle
dictionary={}
dictionary["H"]=Hs
dictionary["S"]=Ss
dictionary["xvals"]=x_of_interest
dictionary["E_EVC"]=E_EVC
dictionary["E_FCI"]=E_exact
dictionary["sample_x"]=sample_x
dictionary["sample_E"]=UCC2_energies
file="energy_data/UCC2_BeH2_stretch.bin"
import pickle
with open(file,"wb") as f:
    pickle.dump(dictionary,f)

ax1.plot(x_of_interest,E_exact,label="FCI",color="b")
ax1.plot(x_of_interest,E_EVC,label="EVC",color="r")
ax1.plot(sample_x,UCC2_energies,"*",label="Sample points",color="m")
ax1.set_title("Potential energy surface")
ax1.set_ylabel("Energy (Hartree)")
ax1.set_xlabel("Interatomic distance (Bohr)")
ax1.legend()
ax2.plot(x_of_interest,abs(np.array(E_EVC)-np.array(E_exact)),label="EVC",color="r")
ax2.plot(sample_x,abs(np.array(UCC2_energies)-np.array(sample_exact)),"*",label="Sample points",color="m")
ax2.set_title(r"Deviation from $E_{FCI}$")
ax2.set_ylabel("Energy (Hartree)")
ax2.set_xlabel("Interatomic distance (Bohr)")
ax2.set_yscale('log')
ax2.legend()
plt.tight_layout()
plt.savefig("Exact_EVC_BeH2_sampleAtOneGeom.pdf")
plt.show()
