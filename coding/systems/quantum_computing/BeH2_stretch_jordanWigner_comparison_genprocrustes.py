import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from quantum_library import *
def molecule(x):
    return "Be 0 0 0; H 0 0 %f; H 0 0 -%f"%(x,x)
file="energy_data/UCC2_BeH2_stretch.bin"
import pickle
with open(file,"rb") as f:
    dicty=pickle.load(f)
E_exact=np.array(dicty["E_FCI"])
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
sample_x=[1.5,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,3.5,4.5,5.5,5.8,6.5]
interest_numbers=list(np.arange(len(sample_x)))
mo_coeff_min_sample=[]
mo_coeff_min=[]
x_of_interest=xvals=np.linspace(1.5,6.5,51)

for k,x in enumerate(sample_x):
    print(x)
    atom=molecule(x)
    mol = gto.M(atom=atom, basis=basis, unit='bohr')
    mo_coeff_temp=[]
    mo_en_temp=[]
    mf = scf.RHF(mol)
    e=mf.kernel(verbose=0)
    mo_coeff_min_sample.append(localize_procrustes(mol,mf.mo_coeff,mf.mo_occ,ref_det,active_orbitals=active_space,nelec=nelec)) #The MO coeff of the lowest lying state
for k,x in enumerate(x_of_interest):
    print(x)
    atom=molecule(x)
    mol = gto.M(atom=atom, basis=basis, unit='bohr')
    mo_coeff_temp=[]
    mo_en_temp=[]
    mf = scf.RHF(mol)
    e=mf.kernel(verbose=0)
    mo_coeff_min.append(localize_procrustes(mol,mf.mo_coeff,mf.mo_occ,ref_det,active_orbitals=active_space,nelec=nelec)) #The MO coeff of the lowest lying state

backend= Aer.get_backend('statevector_simulator')

E_EVC=np.zeros(len(x_of_interest))
backend= Aer.get_backend('statevector_simulator')
seed=np.random.randint(100000)
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

basis="STO-6G"
dicterino=loadmat("data/BeH2_Jordanwigner_UCCSD2.mat")
param_list=dicterino["circuits"]
UCCSD_energies=dicterino["UCCSD2"][0]
qubit_converter_nosymmetry = QubitConverter(mapper=JordanWignerMapper(),two_qubit_reduction=False)
qubit_converter_symmetry=qubit_converter_nosymmetry
hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,3,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
qubit_hamiltonian=qubit_hamiltonian.reduce()
num_qubits=qubit_hamiltonian.num_qubits
UCC_circuits=[]
for indexerino,sample_x_val in enumerate(sample_x): #Get circuits
    print("Baemp")
    idx = (np.abs(xvals - sample_x_val)).argmin() #Find the correct index
    x=xvals[idx]
    assert(abs(x-sample_x_val)<1e-5)
    UCC_param=param_list[idx]
    print(UCC_param)

    UCC_ansatz_f,initial_point=UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=1,generalized=False)
    circuit_qi=UCC_ansatz_f.assign_parameters(UCC_param)
    procrustes_state,R=localize_procrustes(mol,mo_coeff_min_sample[indexerino],mf.mo_occ,ref_det,mix_states=True,return_R=True,active_orbitals=active_space,nelec=nelec)
    R=np.linalg.inv(mo_coeff_min_sample[indexerino])@procrustes_state
    R=R[1:,1:]
    print(R)
    num_qubits=qubit_hamiltonian.num_qubits
    qc=QuantumCircuit(num_qubits)
    qc.append(circuit_qi,list(np.arange(num_qubits))) #qc in canonical basis
    gate=get_transformation_circuit(R,active_space,nelec)
    qc.append(gate,list(np.arange(len(active_space)))) #QC in generalized procrustes basis (alpha spin)
    qc.append(gate,list(np.arange(len(active_space),2*len(active_space)))) #QC in generalized procrustes basis (beta spin)
    UCC_circuits.append(qc.decompose()) #Create UCC circuit and add to list
print("Done getting ciruicts")
for i in range(len(UCC_circuits)):
    newcirc=transpile(UCC_circuits[i],backend=backend,optimization_level=2) #Effectivize circuit
    UCC_circuits[i]=newcirc
print("Done optimizing Circuits")


zero1states={}
pauli_string_exp_vals={}
overlaps={}
for i in range(len(sample_x)):
    for j in range(i,len(sample_x)):
        zero1states["%d%d"%(i,j)]=get_01_state(UCC_circuits[i],UCC_circuits[j],num_qubits,backend)
print("Done getting 01 states")

S=np.zeros((len(sample_x),len(sample_x)))
for i in range(len(sample_x)):
    for j in range(i,len(sample_x)):
        S[i,j]=S[j,i]=get_overlap_expectations(zero1states["%d%d"%(i,j)],num_qubits,qi)
print(S)
sample_E=[]
for k,x in enumerate(sample_x):
    hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,x,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
    qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
    h=get_energy_expectations_01state(zero1states["%d%d"%(k,k)],num_qubits,qubit_hamiltonian,qi)
    sample_E.append(h+nuc_rep)
Hs=[]
Ss=[]
for k,x in enumerate(x_of_interest):
    print(k)
    H=np.zeros((len(sample_x),len(sample_x)))
    procrustes_state,R=localize_procrustes(mol,mo_coeff_min[k],mf.mo_occ,ref_det,mix_states=True,return_R=True,active_orbitals=active_space,nelec=nelec)
    R=R[1:,1:]
    hamiltonian,num_particles,num_spin_orbitals,nuc_rep,newGroup=get_basis_Hamiltonian(molecule,x,qi,procrustes_state,basis=basis,active_space=active_space,nelec=nelec)
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
    print(E_EVC[k],E_exact[k])
import pickle
dictionary={}
dictionary["H"]=Hs
dictionary["S"]=Ss
dictionary["xvals"]=x_of_interest
dictionary["E_EVC"]=E_EVC
dictionary["E_FCI"]=E_exact
dictionary["sample_x"]=sample_x
dictionary["sample_E"]=sample_E
file="energy_data/UCC_BeH2_stretch_JordanWigner_genpPocrustes.bin"
import pickle
with open(file,"wb") as f:
    pickle.dump(dictionary,f)
