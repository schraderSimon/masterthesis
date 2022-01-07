from pyscf import gto, scf, ao2mo, fci
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy, ParticleNumber
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

from qiskit.algorithms.optimizers import COBYLA, SPSA,SLSQP
from qiskit_nature.circuit.library import UCC,UCCSD, HartreeFock
from qiskit_nature.algorithms import VQEUCCFactory
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper,BravyiKitaevMapper, BravyiKitaevSuperFastMapper
from qiskit.utils import QuantumInstance
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.opflow import X, Y, Z, I
from qiskit.opflow import CircuitStateFn, StateFn, TensoredOp
from qiskit.opflow import PauliExpectation, CircuitSampler, StateFn, AerPauliExpectation, MatrixExpectation

mol = mol = gto.M(atom='H 0 0 0; H 0 0 2.0', basis='6-31G',unit="Bohr")
mol.build()
mf = scf.RHF(mol)
mf.kernel()
hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
nuc_rep=mf.energy_nuc()
nuc_rep=0
hcore_mo = np.einsum('pi,pq,qj->ij', mf.mo_coeff, hcore_ao, mf.mo_coeff)
eri_ao = mol.intor('int2e')
eri_mo = ao2mo.incore.full(eri_ao, mf.mo_coeff)
cisolver = fci.FCI(mf)
print("Energy using pyscf. (exact):", cisolver.kernel()[0])



one_body_ints = hcore_mo
two_body_ints = eri_mo
electronic_energy = ElectronicEnergy.from_raw_integrals(
    ElectronicBasis.MO, one_body_ints, two_body_ints
)
hamiltonian = electronic_energy.second_q_ops()[0]
num_particles=[1,1]
num_spin_orbitals=2*len(hcore_ao)
qubit_converter = QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True)
qubit_op = qubit_converter.convert(hamiltonian,num_particles=num_particles)

#print(qubit_op)
backend=Aer.get_backend("aer_simulator")
seed=100
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

numPyEigensolver=NumPyEigensolver()
result = numPyEigensolver.compute_eigenvalues(qubit_op)
print("Energy using qiskit (exact):", result.eigenvalues[0]+nuc_rep)

initial_state = HartreeFock(
        num_spin_orbitals=num_spin_orbitals,
        num_particles=num_particles,
        qubit_converter=qubit_converter,
    )
reps=1
var_form = UCC(
    excitations="sd",
    num_particles=num_particles,
    num_spin_orbitals=num_spin_orbitals,
    initial_state=initial_state,
    qubit_converter=qubit_converter,
    reps=reps,
)
optimizer=SPSA(maxiter=1000)
optimizer2=COBYLA(maxiter=1000)
from qiskit.tools.visualization import circuit_drawer
vqe = VQE(ansatz=var_form,include_custom=True, optimizer=optimizer,quantum_instance=qi)
vqe_result =vqe.compute_minimum_eigenvalue(qubit_op)
vqe2=VQE(ansatz=var_form,include_custom=True, optimizer=optimizer2,quantum_instance=qi)
vqe_result2 =vqe2.compute_minimum_eigenvalue(qubit_op)
print(f'VQE on Aer qasm simulator (no noise): {vqe_result2.eigenvalue.real:.8f}')
print(f'VQE on Aer qasm simulator (no noise): {vqe_result.eigenvalue.real:.8f}')
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.converters import circuit_to_gate
circuit=vqe.get_optimal_circuit()
circuit2=vqe2.get_optimal_circuit()

unitary1=circuit_to_gate(circuit)
unitary2=circuit_to_gate(circuit2)
def get_circuit(hamiltonian,ansatz,optimizer,qi,include_custom=True):
    pass
def create_ansatz(onebody_op,twobody_op,nelec,num_spin_orbitals,qubit_converter=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True)):
    pass
def calculate_energy_overlap(unitary1,unitary2,num_qubits,hamiltonian,qi,include_custom=True,complex=False):
    qc=QuantumCircuit()
    qr=QuantumRegister(num_qubits+1,"q")
    controlled_unitary1=unitary1.control(1)
    controlled_unitary2=unitary2.control(1)
    qc.add_register( qr )
    qc.h(0)
    qc.x(0)
    qc.append(controlled_unitary1,qr)
    qc.x(0)
    qc.append(controlled_unitary2,qr)
    state=CircuitStateFn(qc)
    overlap_measurement=I
    for i in range(num_qubits-1):
        overlap_measurement=overlap_measurement^I
    if complex==False:
        hamiltonian_to_measure=hamiltonian^(X)
        overlap_measurement=overlap_measurement^X
    else:
        hamiltonian_to_measure=hamiltonian^(X+1j*Y)
        overlap_measurement=overlap_measurement^(X+1j*Y)
    measurable_expression = StateFn(hamiltonian_to_measure, is_measurement=True).compose(state)
    measurable_overlap=StateFn(overlap_measurement, is_measurement=True).compose(state)
    if include_custom:
        expectation_energy = AerPauliExpectation().convert(measurable_expression)
        expectation_overlap=AerPauliExpectation().convert(measurable_overlap)
    else:
        expectation_energy = PauliExpectation().convert(measurable_expression)
        expectation_overlap=PauliExpectation().convert(measurable_overlap)
    sampler_energy = CircuitSampler(qi).convert(expectation_energy)
    sampler_overlap = CircuitSampler(qi).convert(expectation_overlap)
    return sampler_energy.eval(),sampler_overlap.eval()
print(calculate_energy_overlap(unitary1,unitary2,circuit.num_qubits,qubit_op,qi,include_custom=True,complex=False))
import sys
sys.exit(1)
