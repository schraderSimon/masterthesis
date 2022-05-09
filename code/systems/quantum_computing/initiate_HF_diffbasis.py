import matplotlib.pyplot as plt
import numpy as np
import openfermion, cirq
from qiskit import QuantumCircuit, transpile
from measure_overlap import *
def molecule(x):
    y = lambda x: 2.54 - 0.46*x
    atom="H  " + str(-y(x)) + " 0 " + str(x) + "; H " + str(y(x)) + " 0  " + str(x) + "; Be 0 0 0"
    return atom
basis="STO-6G"
ref_x=1
mol = mol = gto.M(atom=molecule(ref_x), basis=basis,unit="Bohr")
mol.build()
nelec=mol.nelec
mf = scf.RHF(mol)
mf.kernel()
hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
nuc_rep=mf.energy_nuc()
ref_det=mf.mo_coeff
mol = mol = gto.M(atom=molecule(4), basis=basis,unit="Bohr")
mol.build()
mf = scf.RHF(mol)
mf.kernel()
hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
nuc_rep=mf.energy_nuc()

new_det=mf.mo_coeff
active_space=[1,2,3,4,5,6]
nelec=4
procrustes_state,R=localize_procrustes(mol,new_det,mf.mo_occ,ref_det,mix_states=True,return_R=True,active_orbitals=active_space,nelec=nelec)

n_qubits=len(active_space)
qubits = cirq.LineQubit.range(n_qubits)
circuit = cirq.Circuit(openfermion.bogoliubov_transform(qubits,R))
circuit_as_unitary=cirq.unitary(circuit)
circuit_as_unitary[abs(circuit_as_unitary)<1e-14]=0
#print(circuit_as_unitary.to_qasm())
a,b=np.nonzero(circuit_as_unitary)

backend= Aer.get_backend('statevector_simulator')
seed=np.random.randint(100000)
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
optimizer=SciPyOptimizer("BFGS")
numPyEigensolver=NumPyEigensolver()

qubit_converter = QubitConverter(mapper=JordanWignerMapper(),two_qubit_reduction=False)
import time
start=time.time()
hamiltonian,num_particles,num_spin_orbitals,nuc_rep,orig_group=get_basis_Hamiltonian(molecule,4,qi,new_det,basis="STO-6G",active_space=active_space,nelec=nelec)

qubit_hamiltonian=qubit_converter.convert(hamiltonian)
#ansatz,init_point=UCC_ansatz(num_particles,num_spin_orbitals,n_qubits,qubit_converter=qubit_converter,reps=1,initial_state=None,generalized=False)
ansatz=HartreeFock(num_spin_orbitals=num_spin_orbitals,num_particles=num_particles,qubit_converter=qubit_converter)
state=CircuitStateFn(ansatz)
measurable_expression = StateFn(qubit_hamiltonian, is_measurement=True).compose(state)
expectation = AerPauliExpectation().convert(measurable_expression)
sampler = CircuitSampler(qi).convert(expectation)
value=sampler.eval()
print(value+nuc_rep)
qc=QuantumCircuit(n_qubits*2)
qc.append(ansatz,list(np.arange(n_qubits*2)))
qc.unitary(circuit_as_unitary,list(np.arange(n_qubits-1,-1,-1)))
qc.unitary(circuit_as_unitary,list(np.arange(n_qubits-1,-1,-1)+n_qubits))
hamiltonian_copy=hamiltonian
hamiltonian, num_particles,num_spin_orbitals, nuc_rep,newGroup=get_basis_Hamiltonian(molecule,4,qi,procrustes_state,basis="STO-6G",active_space=None,nelec=None)
qubit_hamiltonian=qubit_converter.convert(hamiltonian)
state1=CircuitStateFn(qc)
measurable_expression = StateFn(qubit_hamiltonian, is_measurement=True).compose(state1)
expectation = AerPauliExpectation().convert(measurable_expression)
sampler = CircuitSampler(qi).convert(expectation)
value=sampler.eval()
print(value+nuc_rep)

"""
To summarize:
- I am setting up the Hamiltonian with respect to the procrustes state. They are as close to the "reference" ones one can get.
- I insert a wave function which is optimized with respect to the reference.
Hence, I first insert the Procrustes-HF state, which is shit, and then transform it into the "real" HF-state by unitarily acting on it with R.
"""
"""
The Procrustes state is a shit state UNDER ITS OWN Hamiltonian.
"""
