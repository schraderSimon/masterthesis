import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from quantum_library import *
def molecule(x):
    y = lambda x: 2.54 - 0.46*x
    atom="H  " + str(-y(x)) + " 0 " + str(x) + "; H " + str(y(x)) + " 0  " + str(x) + "; Be 0 0 0"
    return atom
basis="STO-6G"
mo_coeff_min_sample=[]
irreps_sample=[]
occdict1={"A1":6,"B1":0,"B2":0}
occdict2={"A1":4,"B1":2,"B2":0}
occdict3={"A1":4,"B1":0,"B2":2}
occdicts=[occdict1,occdict2,occdict3]
sample_x=np.array([0,1,1.5,2,2.5,3,3.5,4])
energies=np.zeros((len(sample_x),3))
energies_min=np.zeros(len(sample_x))
active_space=[1,2,3,4,5,6]
nelec=4
x_EVC=np.linspace(0,4,81)
for k,x in enumerate(sample_x):
    print(x)
    atom=molecule(x)
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
    mo_coeff_temp=[]
    mo_en_temp=[]
    for i in [0,1,2]:
        mf = scf.RHF(mol)
        mf.verbose=0
        mf.irrep_nelec=occdicts[i]
        e=mf.kernel(verbose=0)
        mo_coeff_temp.append(mf.mo_coeff)
        mo_en_temp.append(mf.mo_energy)
        energies[k,i]=e
    emindex=np.argmin(energies[k,:])
    irreps_sample.append(occdicts[emindex]) #The occdict of the lowest lying state.
    mo_coeff_min_sample.append(mo_coeff_temp[emindex]) #The MO coeff of the lowest lying state
backend= Aer.get_backend('statevector_simulator')
seed=np.random.randint(100000)
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

optimizer=SciPyOptimizer("BFGS")
eigsolv_k=5
numPyEigensolver=NumPyEigensolver(k=eigsolv_k)

qubit_converter= QubitConverter(mapper=JordanWignerMapper())
optimal_params=None
qasm_strings=[]
from qiskit import execute, Aer
print("enter try")
#sample_x=np.array([0,0.1,0.2,0.3,0.4,1,2,2.5,3,4])
interest_numbers=[0,1,2,3,4,5,6] #All 6 of them :)
try:
    raise FileNotFoundError
    infile=loadmat("data/BeH2_insertion_FCI_qasms.mat")
    sample_x=infile["xvals"][0][interest_numbers]
    qasm_strings=infile["qasm_strings"][interest_numbers] #Read in qasm strings
    print("Done reading circuits")

    hamiltonian,num_particles,num_spin_orbitals,nuc_rep,orig_group=get_basis_Hamiltonian(molecule,x,qi,mo_coeff_min_sample[0],basis="STO-6G",active_space=active_space,nelec=nelec,symmetry='C2v',irreps=irreps_sample[0])
    qubit_hamiltonian=qubit_converter.convert(hamiltonian,num_particles=num_particles)
    num_qubits=qubit_hamiltonian.num_qubits
except FileNotFoundError:
    #If file is not foudn, create qasms
    eigenvectors=[]
    eigenvalues=np.zeros(len(sample_x))
    unitaries=[]
    for k,x in enumerate(sample_x):
        print(k)
        hamiltonian,num_particles,num_spin_orbitals,nuc_rep,orig_group=get_basis_Hamiltonian(molecule,x,qi,mo_coeff_min_sample[k],basis="STO-6G",active_space=active_space,nelec=nelec,symmetry='C2v',irreps=irreps_sample[k])
        qubit_hamiltonian=qubit_converter.convert(hamiltonian,num_particles=num_particles)
        num_qubits=qubit_hamiltonian.num_qubits
        result = numPyEigensolver.compute_eigenvalues(qubit_hamiltonian) #Eigenvectors w.r.t. to canonical orbitals!!
        eigenvalues_k=np.real(result.eigenvalues)+nuc_rep
        if np.abs(eigenvalues_k[0]-eigenvalues_k[1])<1e-10: #The triplet state with...triple degenerate energy
            eigenvalues[k]=(eigenvalues_k[3])
            eigenvectors.append(result.eigenstates[3])
        else:
            eigenvalues[k]=(eigenvalues_k[0])
            eigenvectors.append(result.eigenstates[0])
        statevec=eigenvectors[-1].primitive
        qc=QuantumCircuit(num_qubits)
        qc.isometry(eigenvectors[-1].primitive,list(np.arange(num_qubits)),[])
        stringy=qc.qasm(formatted=False)
        qasm_strings.append(stringy)
    dicterino={}
    dicterino["xvals"]=sample_x
    dicterino["qasm_strings"]=qasm_strings
    savemat("data/BeH2_insertion_FCI_qasms_new.mat",dicterino)
print("exit try")
circuits=[]
ref_x=0
mol = mol = gto.M(atom=molecule(ref_x), basis=basis,unit="Bohr")
mol.build()
mf = scf.RHF(mol)
mf.kernel()
hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
nuc_rep=mf.energy_nuc()
ref_det=mf.mo_coeff

#Enumerate over samples to obtain the Full-CI circuits
for k,sample_x_val in enumerate(sample_x): #Get circuits
    idx = (np.abs(sample_x - sample_x_val)).argmin()
    x=sample_x[idx]
    circuit_qi=QuantumCircuit.from_qasm_str(qasm_strings[idx]) #Read in circuit

    #Find R to transform sample mo's to procrustes MOs
    procrustes_state,R=localize_procrustes(mol,mo_coeff_min_sample[interest_numbers[idx]],mf.mo_occ,ref_det,mix_states=True,return_R=True,active_orbitals=active_space,nelec=nelec)
    R=np.linalg.inv(mo_coeff_min_sample[interest_numbers[idx]])@procrustes_state
    R=R[1:,1:]
    num_qubits=qubit_hamiltonian.num_qubits
    qc=QuantumCircuit(num_qubits)
    qc.append(circuit_qi,list(np.arange(num_qubits))) #qc in canonical basis
    gate=get_transformation_circuit(R,active_space,nelec)

    qc.append(gate,list(np.arange(len(active_space)))) #QC in generalized procrustes basis (alpha spin)
    qc.append(gate,list(np.arange(len(active_space),2*len(active_space)))) #QC in generalized procrustes basis (beta spin)

    circuits.append(qc.decompose())
print("Done getting ciruicts")
zero1states={}
pauli_string_exp_vals={}
overlaps={}
for i in range(len(sample_x)):
    for j in range(i,len(sample_x)):
        print("Getting circuit %f %f"%(sample_x[i],sample_x[j]))
        zero1states["%d%d"%(i,j)]=get_01_state(circuits[i],circuits[j],num_qubits,backend)
print("Done getting new states")
print(zero1states)



irreps=[]
mo_coeff_min=[]
x_of_interest=np.concatenate((np.linspace(0,2.5,26),np.linspace(2.6,3.0,13),np.linspace(3.1,4.0,10)))
energies=np.zeros((len(x_of_interest),3))
for k,x in enumerate(x_of_interest):
    print(x)
    atom=molecule(x)
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
    mo_coeff_temp=[]
    mo_en_temp=[]
    for i in [0,1,2]:
        mf = scf.RHF(mol)
        mf.verbose=0
        mf.irrep_nelec=occdicts[i]
        e=mf.kernel(verbose=0)
        mo_coeff_temp.append(mf.mo_coeff)
        mo_en_temp.append(mf.mo_energy)
        energies[k,i]=e
    emindex=np.argmin(energies[k,:])
    irreps.append(occdicts[emindex])
    mo_coeff_min.append(mo_coeff_temp[emindex])


E_EVC=np.zeros(len(x_of_interest))
E_exact=np.zeros(len(x_of_interest))
S=np.zeros((len(sample_x),len(sample_x)))
for i in range(len(sample_x)):
    for j in range(i,len(sample_x)):
        S[i,j]=S[j,i]=get_overlap_expectations(zero1states["%d%d"%(i,j)],num_qubits,backend)
E_FCI=np.zeros(len(x_of_interest))
print(np.linalg.eig(S)[0])
for k,x in enumerate(x_of_interest):
    atom=molecule(x)
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
    mf = scf.RHF(mol)
    e=mf.kernel(verbose=0)
    cisolver = fci.FCI(mf)
    E_FCI[k]=cisolver.kernel(verbose=0)[0]
diagonal_elements=[]
for k,x in enumerate(x_of_interest):
    print(k)
    H=np.zeros((len(sample_x),len(sample_x)))
    procrustes_state,R=localize_procrustes(mol,mo_coeff_min[k],mf.mo_occ,ref_det,mix_states=True,return_R=True,active_orbitals=active_space,nelec=nelec)
    R=R[1:,1:]
    procrastes_hamiltonian,num_particles,num_spin_orbitals,nuc_rep,newGroup=get_basis_Hamiltonian(molecule,x,qi,procrustes_state,basis=basis,active_space=active_space,nelec=nelec,symmetry='C2v',irreps=irreps[k])
    qubit_hamiltonian=qubit_converter.convert(procrastes_hamiltonian).reduce() #Hamiltonian in Procrustes form.
    for i in range(len(sample_x)):
        for j in range(i,len(sample_x)):
            s=S[i,j]
            h=get_energy_expectations_01state(zero1states["%d%d"%(i,j)],num_qubits,qubit_hamiltonian,backend)
            H[i,j]=H[j,i]=np.real(h+s*nuc_rep)
            print(h,s)

            if (i==j and np.abs(sample_x[i]-x)<1e-5):
                diagonal_elements.append(H[i,j]) #Lazy way to get sample points
    e,cl,c=eig(scipy.linalg.pinv(S,atol=1e-8)@H,left=True) #lazy way to solve generalized eigenvalue problem
    idx = np.real(e).argsort()
    e = e[idx]
    c = c[:,idx]
    cl = cl[:,idx]
    E_EVC[k]=np.real(e[0])
    print(E_EVC[k],E_FCI[k])
dictionary={}
dictionary["xvals"]=x_of_interest
dictionary["E_EVC"]=E_EVC
dictionary["E_FCI"]=E_EVC
dictionary["sample_x"]=sample_x
dictionary["sample_energy"]=diagonal_elements
file="energy_data/FCI_BeH2_insertion.bin"
import pickle
with open(file,"wb") as f:
    pickle.dump(dictionary,f)

plt.plot(x_of_interest,E_EVC,label="EVC (FCI)")
plt.plot(x_of_interest,E_FCI,label="FCI")
try:
    plt.plot(sample_x,diagonal_elements,"*",label="sample points")
except:
    pass

plt.legend()
plt.ylabel("Energy (Hartree)")
plt.xlabel("x (Bohr)")
plt.tight_layout()
plt.savefig("BeH2_difficult_FCI.pdf")
plt.show()
