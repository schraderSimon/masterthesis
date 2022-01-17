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
from qiskit.opflow.primitive_ops import Z2Symmetries
import sys
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
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
sample_x=[2,3,4,5,6]
x_of_interest=np.linspace(6.5,1.5,51)
E_EVC=np.zeros(len(x_of_interest))
E_exact=np.zeros(len(x_of_interest))
E_UCC=np.zeros(len(x_of_interest))
E_k2=np.zeros(len(x_of_interest))
backend= Aer.get_backend('statevector_simulator')
seed=np.random.randint(100000)
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

optimizer=SciPyOptimizer("BFGS")#L_BFGS_B()
numPyEigensolver=NumPyEigensolver()

qubit_converter_nosymmetry = QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True)
#tapering_value=find_symmetry(molecule,ref_x,qi,qubit_converter_nosymmetry,active_space,nelec,basis)
tapering_value=[1,1,1]
qubit_converter_symmetry=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True,z2symmetry_reduction=tapering_value)
UCC_3UPCCG_params=loadmat("BeH2_UCC_vals_extrak2.mat")
UCC_UCC2_params=loadmat("BeH2_UCC_vals.mat")
UCC_1_params=UCC_UCC2_params["UCC_1"]
UCC_2_params=UCC_UCC2_params["UCC_2"]
UCC_3UP_params=UCC_3UPCCG_params["3-UPGCC"]
xvals=UCC_3UPCCG_params["xvals"][0]
UCC_circuits=[]
kUPCC_circuits=[]
UCC_2_circuits=[]
hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,1,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
num_qubits=qubit_hamiltonian.num_qubits
for sample_x_val in sample_x: #Get circuits
    print("Baemp")
    idx = (np.abs(xvals - sample_x_val)).argmin()
    x=xvals[idx]
    UCC_param=UCC_1_params[idx]
    UCC_2_param=UCC_2_params[idx]
    UCC_3UP_param=UCC_3UP_params[idx]
    UCC_ansatz_f,initial_point=UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=1,generalized=False)
    UCC_2_ansatz_f,initial_point=UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=2,generalized=False)
    ansatz_k3_f,initial_point_k2=kUpUCCSD_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=3)
    UCC_circuits.append(UCC_ansatz_f.assign_parameters(UCC_param))
    UCC_2_circuits.append(UCC_2_ansatz_f.assign_parameters(UCC_2_param))
    kUPCC_circuits.append(ansatz_k3_f.assign_parameters(UCC_3UP_param))

print("Done getting ciruicts")
import time
optimization_levels=[0,1,2,3]
print("")
idiots=[UCC_circuits[0],UCC_2_circuits[0],kUPCC_circuits[0]]
names=["Standard UCC", "UCC_2", "kUPGCC"]
for i in range(3):
    circuit_to_transpile=idiots[i]
    print(names[i])
    for optimization_level in optimization_levels:
        start = time.perf_counter()
        newcirc=transpile(circuit_to_transpile,backend=backend,optimization_level=optimization_level)
        end = time.perf_counter()
        print("UCC_2Opt-lev: %d, Time: %f, Num_nonlocal_gates: %d,unitary factors: %d "%(optimization_level,end-start,newcirc.num_nonlocal_gates(),newcirc.num_unitary_factors()))
        newUn=circuit_to_gate(newcirc)
        start = time.perf_counter()
        h,s=calculate_energy_overlap(newUn,newUn,num_qubits,qubit_hamiltonian,qi,nuc_rep,include_custom=True,complex=False)
        end = time.perf_counter()
        print("Time to compute h,s:%f"%(end-start))
