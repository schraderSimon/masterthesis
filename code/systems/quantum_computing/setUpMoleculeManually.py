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
from qiskit import Aer

from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit_nature.circuit.library import UCC,UCCSD, HartreeFock
from qiskit_nature.algorithms import VQEUCCFactory
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper,BravyiKitaevMapper, BravyiKitaevSuperFastMapper

mol = mol = gto.M(atom='H 0 0 0; H 0 0 2.0', basis='STO-6G',unit="Bohr")
mol.build()
mf = scf.RHF(mol)
mf.kernel()
hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
nuc_rep=mf.energy_nuc()
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
backend=Aer.get_backend("aer_simulator_statevector")
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
optimizer=COBYLA(maxiter=1000)
from qiskit.tools.visualization import circuit_drawer

vqe = VQE(ansatz=var_form,include_custom=True, optimizer=optimizer,quantum_instance=Aer.get_backend("aer_simulator_statevector"))
measurement=vqe.construct_expectation([0,0,0.2],qubit_op) #Expectation circuit of a measurement of the qubit-op uing the parameters 0, 0, 0.2
print(measurement)
vqe_result =vqe.compute_minimum_eigenvalue(qubit_op)
print(vqe_result)
print("UCCSD using qiskit (noexact)", np.real(vqe_result.eigenvalue)+nuc_rep)
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

circuit=vqe.get_optimal_circuit()
circuit2=vqe.get_optimal_circuit()

#anc = QuantumRegister(1, 'ancilla')
#cr = QuantumRegister(4, 'c')
#qc = QuantumCircuit(cr,anc)
circuit=circuit.compose(circuit2) #Combine two circuits (this will I need later!)
circuit_drawer(circuit2.decompose(), output='mpl',initial_state=True)
plt.show()
energy_func=vqe.get_energy_evaluation(qubit_op) #Returns a function of the energy expectation value as function of the qubit operator
def energy_funcnew(input):
    return energy_func(input)+nuc_rep
from scipy.optimize import minimize

res=minimize(energy_funcnew,[1,2,3]*reps)
print("UCCSD using scipy (noexact):",energy_funcnew(res.x))
from qiskit.opflow import PauliExpectation, CircuitSampler, StateFn, AerPauliExpectation, MatrixExpectation

expectation = PauliExpectation().convert(measurement)
sampler = CircuitSampler(backend).convert(expectation)
print('Snapshot:', sampler.eval().real)

expectation = MatrixExpectation().convert(measurement)
sampler = CircuitSampler(backend).convert(expectation)
print('Matrix:', sampler.eval().real)
