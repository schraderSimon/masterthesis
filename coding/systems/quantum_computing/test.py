from qiskit_nature.drivers.second_quantization.pyscfd import PySCFDriver
from pyscf import gto, scf, ao2mo, fci
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
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem

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
from qiskit.algorithms.optimizers import COBYLA, SPSA,SLSQP
from qiskit_nature.circuit.library import UCC,UCCSD, HartreeFock
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
from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer
from qiskit.opflow.primitive_ops import Z2Symmetries
from qiskit.opflow import PauliSumOp, TaperedPauliSumOp
import sys
import warnings
electronic_driver = PySCFDriver(atom="Li 0 0 0; H 0 0 4",basis="STO-6G")
electronic_driver_result = electronic_driver.run()
electronic_energy=electronic_driver_result.get_property("ElectronicEnergy")
twobody=electronic_energy.get_electronic_integral(ElectronicBasis.MO,2)
particle_number=electronic_driver_result.get_property("ParticleNumber")
basistransform=electronic_driver_result.get_property("ElectronicBasisTransform")
grouped_property=ElectronicStructureDriverResult()
grouped_property.add_property(electronic_energy)
grouped_property.add_property(particle_number)
grouped_property.add_property(basistransform)
active_space_trafo = ActiveSpaceTransformer(
    2, 3, [1,2,4]
)
qubit_converter = QubitConverter(mapper=ParityMapper(),z2symmetry_reduction = "auto",two_qubit_reduction=True)

problem = ElectronicStructureProblem(electronic_driver)#, transformers=[active_space_trafo])
#print(problem._grouped_property_transformed)
qubit_op = qubit_converter.convert(problem.second_q_ops()[0],num_particles=4)
newGroup=active_space_trafo.transform(grouped_property)
hamiltonian = newGroup.second_q_ops()[0]

print(hamiltonian)
hamiltonian = qubit_converter.convert(hamiltonian,num_particles=2)
result = NumPyEigensolver().compute_eigenvalues(hamiltonian)
#print(result)

pauli_symm = Z2Symmetries.find_Z2_symmetries(hamiltonian)
print(qubit_op.num_qubits)
print(pauli_symm)
print(qubit_op)
qubit_op_new=pauli_symm.taper(hamiltonian).reduce()
print(qubit_op_new)
for qubit_oppy in qubit_op_new:
    print(qubit_oppy.num_qubits)
    result = NumPyEigensolver().compute_eigenvalues(qubit_oppy)
    print(result.eigenvalues[0])
sys.exit(1)

problem2 = ElectronicStructureProblem(electronic_driver)
qubit_op = qubit_converter.convert(problem2.second_q_ops()[0],num_particles=6)
print(qubit_op)
