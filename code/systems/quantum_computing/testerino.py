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
from scipy.io import loadmat, savemat
import sys
import warnings
from qiskit.opflow.list_ops import ListOp
warnings.filterwarnings('ignore', category=DeprecationWarning)
backend= Aer.get_backend('aer_simulator')
shots=1000
qi=QuantumInstance(backend=backend,shots=shots)
qc=QuantumCircuit(1,0)
state=CircuitStateFn(qc)
measurable_expression = StateFn(ListOp((X,Z,Y)), is_measurement=True).compose(state)
pauliExpec=PauliExpectation()
expectation = pauliExpec.convert(measurable_expression)
sampler = CircuitSampler(qi,param_qobj=True)
postSampler=sampler.convert(expectation)
expectations=postSampler.eval()
ups=(1+np.array(expectations))/2
downs=(1-np.array(expectations))/2
print(ups)
print(downs)
