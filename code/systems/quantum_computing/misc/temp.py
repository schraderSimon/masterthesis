import sys
sys.path.append("../../libraries")
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
x_of_interest=xvals
sample_x=x_of_interest
E_EVC=np.zeros(len(x_of_interest))
E_exact=np.zeros(len(x_of_interest))
E_UCC=np.zeros(len(x_of_interest))

print(xvals)
qubit_converter_nosymmetry = QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True)
tapering_value=[1,1,1]
qubit_converter_symmetry=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True,z2symmetry_reduction=tapering_value)

UCC_circuits=[]

#All of this just to get the number of qubits...
hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,3,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
qubit_hamiltonian=qubit_hamiltonian.reduce()
num_qubits=qubit_hamiltonian.num_qubits
for idx,x in enumerate(xvals): #Get circuits
    print("Baemp")
    print(x)
    UCC_param=UCC_1_params[idx]
    print(num_particles,num_spin_orbitals,num_qubits,)
    UCC_ansatz_f,initial_point=UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=1,generalized=False)
    print(len(initial_point))
    print(len(UCC_param))
    UCC_circuits.append(UCC_ansatz_f.assign_parameters(UCC_param)) #Create UCC circuit and add to list
for i in range(len(UCC_circuits)):
    newcirc=transpile(UCC_circuits[i],backend=backend,optimization_level=1) #Effectivize circuit
    UCC_circuits[i]=newcirc
print("Done transforming circuits")
zero1states={}
pauli_string_exp_vals={}
overlaps={}
for i in range(len(sample_x)):
    zero1states["%d%d"%(i,i)]=get_01_state(UCC_circuits[i],UCC_circuits[i],num_qubits,backend)
print("Done getting new states")
energies=[]
for k,x in enumerate(xvals):
    print(k)
    hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,x,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
    qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
    h=get_energy_expectations_01state(zero1states["%d%d"%(k,k)],num_qubits,qubit_hamiltonian,qi)
    energies.append(np.real(h+nuc_rep))
    print(energies[-1])
print("UCC 1 energies:")
print(list(energies))
