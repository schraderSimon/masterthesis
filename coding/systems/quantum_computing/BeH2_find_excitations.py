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
tapering_value=find_symmetry(molecule,ref_x,qi,qubit_converter_nosymmetry,active_space,nelec,basis)
qubit_converter_symmetry=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True,z2symmetry_reduction=tapering_value)

optimal_params=None
optimal_params_k2=None
UCC_1_params=[]
UCC_2_params=[]

for k,x in enumerate(x_of_interest):
    hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,x,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
    qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
    num_qubits=qubit_hamiltonian.num_qubits
    result = numPyEigensolver.compute_eigenvalues(qubit_hamiltonian)
    E_exact[k]=np.real(result.eigenvalues[0]+nuc_rep)
    ansatz_k2,initial_point_k2=kUpUCCSD_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=3)
    if optimal_params_k2 is None:
        optimal_params_k2=initial_point_k2
    unitary,energy_k2,optimal_params_k2=get_unitary(qubit_hamiltonian,ansatz_k2,optimizer,qi,include_custom=True,initial_point=optimal_params_k2,nuc_rep=nuc_rep)
    UCC_2_params.append(optimal_params_k2)
    E_k2[k]=np.real(energy_k2)
    ansatz,initial_point=UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=1,generalized=False)
    if optimal_params is None:
        optimal_params=initial_point
    unitary,energy,optimal_params=get_unitary(qubit_hamiltonian,ansatz,optimizer,qi,include_custom=True,initial_point=optimal_params,nuc_rep=nuc_rep)
    UCC_1_params.append(optimal_params)
    E_UCC[k]=np.real(energy)
    print("Exact: %f, UCC: %f, k2: %f"%(E_exact[k],E_UCC[k],E_k2[k]))
dicterino={}
dicterino["xvals"]=x_of_interest
dicterino["UCC_1"]=UCC_1_params
dicterino["3-UPGCC"]=UCC_2_params
savemat("BeH2_UCC_vals_extrak2.mat",dicterino)
plt.plot(x_of_interest,E_exact,"-",label="exact diagonalization")
plt.plot(x_of_interest,E_UCC,"-",label="UCC")
plt.plot(x_of_interest,E_k2,"-",label="k3")
plt.legend()
plt.show()