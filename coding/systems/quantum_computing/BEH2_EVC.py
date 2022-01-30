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
from qiskit.opflow.primitive_ops import Z2Symmetries, PauliOp
import sys
import warnings
from scipy.io import loadmat, savemat

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
sample_x=[6.5,5.5,4.5,3.5,2.5,1.5]
#sample_x=[5.5]
x_of_interest=np.linspace(6.5,1.5,51)
E_EVC=np.zeros(len(x_of_interest))
E_exact=np.zeros(len(x_of_interest))
E_UCC=np.zeros(len(x_of_interest))
E_k2=np.zeros(len(x_of_interest))
backend= AerSimulator(method='statevector',max_parallel_threads=4)
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
hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,3,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
qubit_hamiltonian=qubit_hamiltonian.reduce()
print(qubit_hamiltonian)
op_list=[]
coeff_list=[]
for pauliOpStr,coeff in qubit_hamiltonian.primitive.to_list():
    op_list.append((PauliOp(Pauli(pauliOpStr))))
    coeff_list.append(coeff)
print("------")
print(operator)
result = NumPyEigensolver().compute_eigenvalues(qubit_hamiltonian)
print("Eigenvalue before shit: %f"%np.real(result.eigenvalues[0]))
result = NumPyEigensolver().compute_eigenvalues(operator)
print("Eigenvalue after shit: %f"%np.real(result.eigenvalues[0]))

sys.exit(1)

print("Len OP list: %d"%len(op_list))
num_qubits=qubit_hamiltonian.num_qubits
for sample_x_val in sample_x: #Get circuits
    print("Baemp")
    idx = (np.abs(xvals - sample_x_val)).argmin()
    x=xvals[idx]
    UCC_param=UCC_1_params[idx]
    UCC_2_param=UCC_2_params[idx]
    UCC_3UP_param=UCC_3UP_params[idx]
    #UCC_ansatz_f,initial_point=UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=1,generalized=False)
    UCC_2_ansatz_f,initial_point=UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=2,generalized=False)
    #ansatz_k3_f,initial_point_k2=kUpUCCSD_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=3)
    #UCC_circuits.append(UCC_ansatz_f.assign_parameters(UCC_param))
    UCC_2_circuits.append(UCC_2_ansatz_f.assign_parameters(UCC_2_param))
    #kUPCC_circuits.append(ansatz_k3_f.assign_parameters(UCC_3UP_param))
print("Done getting ciruicts")
for i in range(len(UCC_circuits)):
    #newcirc=transpile(UCC_circuits[i],backend=backend,optimization_level=1)
    #UCC_circuits[i]=newcirc
    newcirc=transpile(UCC_2_circuits[i],backend=backend,optimization_level=1)
    UCC_2_circuits[i]=newcirc
    #newcirc=transpile(kUPCC_circuits[i],backend=backend,optimization_level=1)
    #kUPCC_circuits[i]=newcirc
print("Done transforming circuits")
zero1states={}
pauli_string_exp_vals={}
overlaps={}
for i in range(len(sample_x)):
    for j in range(i,len(sample_x)):
        zero1states["%d%d"%(i,j)]=get_01_state(UCC_2_circuits[i],UCC_2_circuits[j],num_qubits,backend)
print("Done getting new states")
outfile=open("aaBeH2_UCC2_sampleXPauliStrings.txt","w")
outfile.write("Sample_x:\n")
for k,x in enumerate(sample_x):
    outfile.write("%d %f\n"%(k,x))
outfile.write("-\n")
for i in range(len(sample_x)):
    for j in range(i,len(sample_x)):
        overlaps["%d%d"%(i,j)]=get_overlap_expectations(zero1states["%d%d"%(i,j)],num_qubits,qi)
        pauli_string_exp_vals["%d%d"%(i,j)]=get_energy_expectations(zero1states["%d%d"%(i,j)],op_list,qi)
        outfile.write("Overlap %d %d: %f\n"%(i,j,overlaps["%d%d"%(i,j)]))
        outfile.write("Pauli strings %d %d:\n"%(i,j))
        for key in pauli_string_exp_vals["%d%d"%(i,j)]:
                outfile.write("%s %f\n"%(key,pauli_string_exp_vals["%d%d"%(i,j)][key]))
        outfile.write("-\n")
outfile.close()
sys.exit(1)
E_EVC=np.zeros(len(x_of_interest))
E_exact=np.zeros(len(x_of_interest))
for k,x in enumerate(x_of_interest):
    print(k)
    H=np.zeros((len(sample_x),len(sample_x)))
    S=np.zeros((len(sample_x),len(sample_x)))
    hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,x,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
    qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
    coefs_strings=qubit_hamiltonian.reduce().primitive.to_list()
    for i in range(len(sample_x)):
        for j in range(i,len(sample_x)):
            h=0
            s=overlaps["%d%d"%(i,j)]
            for pauliOpStr,coeff in qubit_hamiltonian.primitive.to_list():
                try:
                    h+=pauli_string_exp_vals["%d%d"%(i,j)][pauliOpStr]*coeff
                except KeyError:
                    pass
            S[i,j]=S[j,i]=np.real(s)
            H[i,j]=H[j,i]=np.real(h+s*nuc_rep)
            print(h,s)
    e,cl,c=eig(scipy.linalg.pinv(S,atol=1e-8)@H,left=True)
    idx = np.real(e).argsort()
    e = e[idx]
    c = c[:,idx]
    cl = cl[:,idx]
    E_EVC[k]=np.real(e[0])
    result = numPyEigensolver.compute_eigenvalues(qubit_hamiltonian)
    E_exact[k]=np.real(result.eigenvalues[0]+nuc_rep)
plt.plot(x_of_interest,E_EVC,label="EVC")
plt.plot(x_of_interest,E_exact,label="exact")
plt.legend()
plt.show()
