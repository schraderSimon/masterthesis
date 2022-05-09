import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, cc,scf, ao2mo,fci, mcscf
import scipy
np.set_printoptions(linewidth=300,precision=10,suppress=True)
from scipy.linalg import block_diag, eig, orth
from numba import jit
from scipy.optimize import minimize, root,newton
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy, ParticleNumber
from qiskit_nature.properties.second_quantization.electronic.electronic_structure_driver_result import ElectronicStructureDriverResult
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis, ElectronicBasisTransform
from qiskit.opflow.list_ops import ListOp
from qiskit_nature.properties import GroupedProperty
from qiskit.algorithms import NumPyEigensolver,VQE
#from qiskit.aqua.algorithms import VQE
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    ElectronicIntegrals,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
    IntegralProperty,
)
from qiskit.extensions import UnitaryGate, Initialize
import openfermion, cirq
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit import Aer, transpile, execute
from qiskit.providers.aer import AerSimulator, QasmSimulator
from qiskit.algorithms.optimizers import *
from qiskit_nature.circuit.library import UCC,UCCSD, HartreeFock, PUCCD, SUCCD
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
import openfermion, cirq
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import Statevector
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
def generalized_eigenvector(T,S,threshold=1e-8,symmetric=True):
    """Solves the generalized eigenvector problem.
    Input:
    T: The symmetric matrix
    S: The overlap matrix
    symmetric: Wether the matrices T, S are symmetric
    Returns: The lowest eigenvalue & eigenvector
    """
    ###The purpose of this procedure here is to remove all very small eigenvalues of the overlap matrix for stability
    s, U=np.linalg.eigh(S) #Diagonalize S (overlap matrix, Hermitian by definition)
    U=np.fliplr(U)
    s=s[::-1] #Order from largest to lowest; S is an overlap matrix, hence we won't
    s=s[s>threshold] #Keep only largest eigenvalues
    spowerminushalf=s**(-0.5) #Take s
    snew=np.zeros((len(U),len(spowerminushalf)))
    sold=np.diag(spowerminushalf)
    snew[:len(s),:]=sold
    s=snew


    ###Canonical orthogonalization
    X=U@s
    Tstrek=X.T@T@X
    if symmetric:
        epsilon, Cstrek = np.linalg.eigh(Tstrek)
    else:
        epsilon, Cstrek = np.linalg.eig(Tstrek)
    idx = epsilon.argsort()[::1] #Order by size (non-absolute)
    epsilon = epsilon[idx]
    Cstrek = Cstrek[:,idx]
    C=X@Cstrek
    lowest_eigenvalue=epsilon[0]
    lowest_eigenvector=C[:,0]
    return lowest_eigenvalue,lowest_eigenvector

def orthogonal_procrustes(mo_new,reference_mo):
    A=reference_mo
    B=mo_new.T
    M=B@A
    U,s,Vt=scipy.linalg.svd(M)
    return U@Vt, 0

def localize_procrustes(mol,mo_coeff,mo_occ,ref_mo_coeff,mix_states=False,active_orbitals=None,nelec=None, return_R=False):
    """Performs the orthgogonal procrustes on the occupied and the unoccupied molecular orbitals.
    ref_mo_coeff is the mo_coefs of the reference state.
    If "mix_states" is True, then mixing of occupied and unoccupied MO's is allowed.
    """
    if active_orbitals is None:
        active_orbitals=np.arange(len(mo_coeff))
    if nelec is None:
        nelec=int(np.sum(mo_occ))
    active_orbitals_occ=active_orbitals[:nelec//2]
    active_orbitals_unocc=active_orbitals[nelec//2:]
    mo_coeff_new=mo_coeff.copy()
    if mix_states==False:
        mo=mo_coeff[:,active_orbitals_occ]
        premo=ref_mo_coeff[:,active_orbitals_occ]
        R1,scale=orthogonal_procrustes(mo,premo)
        mo=mo@R1
        mo_unocc=mo_coeff[:,active_orbitals_unocc]
        premo=ref_mo_coeff[:,active_orbitals_unocc]
        R2,scale=orthogonal_procrustes(mo_unocc,premo)
        mo_unocc=mo_unocc@R2


        mo_coeff_new[:,active_orbitals_occ]=np.array(mo)
        mo_coeff_new[:,active_orbitals_unocc]=np.array(mo_unocc)
        R=block_diag(R1,R2)
    elif mix_states==True:
        mo=mo_coeff[:,active_orbitals]
        premo=ref_mo_coeff[:,active_orbitals]
        R,scale=orthogonal_procrustes(mo,premo)
        mo=mo@R

        mo_coeff_new[:,active_orbitals]=np.array(mo)

    if return_R:
        return mo_coeff_new,R
    else:
        return mo_coeff_new

def molecule(x):
    return "Be 0 0 0; H 0 0 %f; H 0 0 -%f"%(x,x)
#molecule=lambda x: "Li 0 0 0; H 0 0 -%f"%x
#molecule=lambda x: "H 0 0 0; H 0 0 2.324362966; H %f 0 0; H %f 0 2.324362966"%(x,x)
def get_basis_Hamiltonian(molecule,x,qi,mo_coeff,basis="STO-6G",active_space=None,nelec=None,symmetry=None,irreps=None):
    mol = mol = gto.M(atom=molecule(x), basis=basis,unit="Bohr",symmetry=symmetry)
    mol.build()
    mf = scf.RHF(mol)
    mf.irrep_nelec=irreps
    eref=mf.kernel()
    hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    nuc_rep=mf.energy_nuc()
    hcore_mo = np.einsum('pi,pq,qj->ij', mo_coeff, hcore_ao, mo_coeff)
    nao = mol.nao_nr()
    eri_ao = mol.intor('int2e')
    eri_mo = ao2mo.incore.full(eri_ao, mo_coeff)
    twobody_MO=TwoBodyElectronicIntegrals(ElectronicBasis.MO, (eri_mo,eri_mo,eri_mo,eri_mo))
    onebody_MO=OneBodyElectronicIntegrals(ElectronicBasis.MO, (hcore_mo,hcore_mo))
    twobody_AO=TwoBodyElectronicIntegrals(ElectronicBasis.AO, (eri_ao,eri_ao,eri_ao,eri_ao))
    onebody_AO=OneBodyElectronicIntegrals(ElectronicBasis.AO, (hcore_ao,hcore_ao))
    electronic_energy=ElectronicEnergy(
            [onebody_AO, twobody_AO, onebody_MO, twobody_MO],
            nuclear_repulsion_energy=nuc_rep,
            reference_energy=eref,
        )
    num_particles=mol.nelec
    num_spin_orbitals=2*len(hcore_ao)
    particle_number = ParticleNumber(
    num_spin_orbitals=num_spin_orbitals,
    num_particles=num_particles,
    )
    basistransform=ElectronicBasisTransform(ElectronicBasis.AO,ElectronicBasis.MO,mo_coeff)
    grouped_property=ElectronicStructureDriverResult()
    grouped_property.add_property(electronic_energy)
    grouped_property.add_property(particle_number)
    grouped_property.add_property(basistransform)
    if active_space is None:
        active_space=np.arange(len(hcore_ao))
    if nelec is None:
        nelec=int(np.sum(mol.nelec))
    transformer= ActiveSpaceTransformer(
        nelec, len(active_space), active_space
    )
    newGroup=transformer.transform(grouped_property)
    energyshift=np.real(newGroup.get_property("ElectronicEnergy")._shift["ActiveSpaceTransformer"])
    hamiltonian = newGroup.second_q_ops()[0]
    num_particles=newGroup.get_property("ParticleNumber").num_particles
    num_spin_orbitals=newGroup.get_property("ParticleNumber").num_spin_orbitals
    return hamiltonian, num_particles,num_spin_orbitals, nuc_rep+energyshift,newGroup

def get_hamiltonian(molecule,x,qi,basis="STO-6G",ref_det=None,active_space=None,nelec=None):

    mol = mol = gto.M(atom=molecule(x), basis=basis,unit="Bohr")
    mol.build()
    mf = scf.RHF(mol)
    eref=mf.kernel()
    hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    nuc_rep=mf.energy_nuc()
    mo_coeff=mf.mo_coeff
    if ref_det is not None:
        mo_coeff=localize_procrustes(mol,mf.mo_coeff,mf.mo_occ,ref_det,active_orbitals=active_space,nelec=nelec)
    hcore_mo = np.einsum('pi,pq,qj->ij', mo_coeff, hcore_ao, mo_coeff)
    nao = mol.nao_nr()
    eri_ao = mol.intor('int2e')
    eri_mo = ao2mo.incore.full(eri_ao, mo_coeff)
    twobody_MO=TwoBodyElectronicIntegrals(ElectronicBasis.MO, (eri_mo,eri_mo,eri_mo,eri_mo))
    onebody_MO=OneBodyElectronicIntegrals(ElectronicBasis.MO, (hcore_mo,hcore_mo))
    twobody_AO=TwoBodyElectronicIntegrals(ElectronicBasis.AO, (eri_ao,eri_ao,eri_ao,eri_ao))
    onebody_AO=OneBodyElectronicIntegrals(ElectronicBasis.AO, (hcore_ao,hcore_ao))
    electronic_energy=ElectronicEnergy(
            [onebody_AO, twobody_AO, onebody_MO, twobody_MO],
            nuclear_repulsion_energy=nuc_rep,
            reference_energy=eref,
        )
    num_particles=mol.nelec
    num_spin_orbitals=2*len(hcore_ao)
    particle_number = ParticleNumber(
    num_spin_orbitals=num_spin_orbitals,
    num_particles=num_particles,
    )
    basistransform=ElectronicBasisTransform(ElectronicBasis.AO,ElectronicBasis.MO,mo_coeff)
    grouped_property=ElectronicStructureDriverResult()
    grouped_property.add_property(electronic_energy)
    grouped_property.add_property(particle_number)
    grouped_property.add_property(basistransform)
    if active_space is None:
        active_space=np.arange(len(hcore_ao))
    if nelec is None:
        nelec=int(np.sum(mol.nelec))
    transformer= ActiveSpaceTransformer(
        nelec, len(active_space), active_space
    )
    newGroup=transformer.transform(grouped_property)
    energyshift=np.real(newGroup.get_property("ElectronicEnergy")._shift["ActiveSpaceTransformer"])
    hamiltonian = newGroup.second_q_ops()[0]
    num_particles=newGroup.get_property("ParticleNumber").num_particles
    num_spin_orbitals=newGroup.get_property("ParticleNumber").num_spin_orbitals
    return hamiltonian, num_particles,num_spin_orbitals, nuc_rep+energyshift
def get_unitary(hamiltonian,ansatz,optimizer,qi,nuc_rep,include_custom=True,initial_point=None):
    vqe = VQE(ansatz=ansatz,include_custom=include_custom, optimizer=optimizer,quantum_instance=qi,initial_point=initial_point)

    vqe_result =vqe.compute_minimum_eigenvalue(hamiltonian)
    circuit=vqe.get_optimal_circuit()
    unitary=circuit_to_gate(circuit)
    optimal_params=vqe.optimal_params
    return unitary, vqe_result.eigenvalue.real+nuc_rep, optimal_params
def k_fold_unitary(num_particles,num_spin_orbitals,num_qubits,qubit_converter,hamiltonian,ansatz_func,ansatz_0,optimizer,qi,include_custom=True,k=3,initial_point=None):
    """Repeated application of the same algorithm"""
    vqe = VQE(ansatz=ansatz_0,include_custom=include_custom, optimizer=optimizer,quantum_instance=qi,initial_point=initial_point)

    vqe_result =vqe.compute_minimum_eigenvalue(hamiltonian)
    print("i=0, E=%f"%(vqe_result.eigenvalue.real+nuc_rep))
    circuit=vqe.get_optimal_circuit()
    unitary=circuit_to_gate(circuit)
    for i in range(1,k):
        new_ansatz,initial_point=ansatz_func(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter,reps=1,initial_state=circuit)
        vqe = VQE(ansatz=new_ansatz,include_custom=include_custom, optimizer=optimizer,quantum_instance=qi,initial_point=initial_point)
        vqe_result =vqe.compute_minimum_eigenvalue(hamiltonian)
        circuit=vqe.get_optimal_circuit()
        unitary=circuit_to_gate(circuit)
        print("i=%d, E=%f"%(i,vqe_result.eigenvalue.real+nuc_rep))
    return unitary, vqe_result.eigenvalue.real+nuc_rep
def SU2_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True),reps=1,initial_state=None):
    if initial_state is None:
        initial_state = HartreeFock(
                num_spin_orbitals=num_spin_orbitals,
                num_particles=num_particles,
                qubit_converter=qubit_converter,
            )
    var_form=EfficientSU2(
            num_qubits=num_qubits,
            su2_gates=None,
            entanglement='circular',
            reps=reps,
            skip_unentangled_qubits=False,
            skip_final_rotation_layer=False,
            parameter_prefix='θ',
            insert_barriers=False,
            initial_state=initial_state,
            name='EfficientSU2',
    )
    init_pt=0.02*(np.random.rand(len(var_form.parameters))-0.5)

    return var_form,init_pt
def UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True),reps=1,initial_state=None,generalized=False):
    if initial_state is None:
        initial_state = HartreeFock(
                num_spin_orbitals=num_spin_orbitals,
                num_particles=num_particles,
                qubit_converter=qubit_converter,
            )
    var_form = UCC(
        excitations="sd",
        num_particles=num_particles,
        num_spin_orbitals=num_spin_orbitals,
        initial_state=initial_state,
        qubit_converter=qubit_converter,
        reps=reps,
        generalized=generalized,
    )
    init_pt=(np.random.rand(len(var_form.parameters))-0.5)
    return var_form, init_pt
def SUCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True),reps=1,initial_state=None,generalized=False,include_singles=True):
    if initial_state is None:
        initial_state = HartreeFock(
                num_spin_orbitals=num_spin_orbitals,
                num_particles=num_particles,
                qubit_converter=qubit_converter,
            )
    var_form = SUCCD(
        num_particles=num_particles,
        num_spin_orbitals=num_spin_orbitals,
        initial_state=initial_state,
        qubit_converter=qubit_converter,
        reps=reps,
        generalized=generalized,
        include_singles=(include_singles,include_singles)
    )
    init_pt=0.1*(np.random.rand(len(var_form.parameters))-0.5)
    return var_form, init_pt

def UPCCSD_excitation_generator(num_particles,num_spin_orbitals):
    num_spin_orbitals=num_spin_orbitals//2
    excitations=[]
    for particle_1 in range(num_spin_orbitals):
        for particle_2 in range(particle_1+1,num_spin_orbitals):
            excitations.append(((particle_1,particle_1+num_spin_orbitals),(particle_2,particle_2+num_spin_orbitals)))
    for particle_1 in range(num_spin_orbitals):
        for particle_2 in range(particle_1+1,num_spin_orbitals):
            excitations.append(((particle_1,),(particle_2,)))
            excitations.append(((particle_1+num_spin_orbitals,),(particle_2+num_spin_orbitals,)))
    return excitations
def kUpUCCSD_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True),reps=1,initial_state=None):
    if initial_state is None:
        initial_state = HartreeFock(
                num_spin_orbitals=num_spin_orbitals,
                num_particles=num_particles,
                qubit_converter=qubit_converter,
            )
    var_form = UCC(
        num_particles=num_particles,
        num_spin_orbitals=num_spin_orbitals,
        initial_state=initial_state,
        qubit_converter=qubit_converter,
        reps=reps,
        excitations=UPCCSD_excitation_generator,
    )
    """
    excitations=UPCCSD_excitation_generator(num_particles,num_spin_orbitals)
    length=len(excitations)
    dictistan={}
    for k in range(reps):
        for i in range(length//3):
            dictistan[var_form.parameters[k*length+2*length//3+i]]=var_form.parameters[k*length+length//3+i]
    var_form.assign_parameters(dictistan, inplace=True)

    print(len(var_form.parameters))
    """
    init_pt=10*0.1*(np.random.rand(len(var_form.parameters))-0.5)
    return var_form, init_pt
get_bin = lambda x, n: format(x, 'b').zfill(n)
def standard_LCU(unitaries,parameters,num_qubits,backend=None):
    number_ancillas=int(np.ceil(np.log2(len(unitaries))))
    number_circuit_qubits=number_ancillas+num_qubits
    qc=QuantumCircuit()
    qr=QuantumRegister(number_circuit_qubits,"q")
    cr=ClassicalRegister(number_ancillas,"c")
    qc.add_register(qr)
    qc.add_register(cr)
    V0=np.pad(np.sqrt(parameters),(0,int(2**number_ancillas-len(parameters))))
    V0=V0/np.linalg.norm(V0)
    c=np.sum(V0**2)
    XL=np.random.rand(len(V0),len(V0))
    XL[:,0]=V0
    V,R=np.linalg.qr(XL)
    if R[0,0]<0:
        V[:,0]=-V[:,0]
    V_gate=UnitaryGate(Operator(V))
    controlled_unitaries=[]
    for k in range(len(unitaries)):
        controlled_unitaries.append(unitaries[k].control(number_ancillas,ctrl_state=get_bin(k,number_ancillas)[::1]))
    qc.append(V_gate,qargs=list(np.arange(0,number_ancillas))[::1])
    print(Statevector(qc))
    print(Statevector(qc).probabilities_dict())
    for k,uni in enumerate(controlled_unitaries):
        qc.append(uni,qr)
    print(Statevector(qc))
    print(Statevector(qc).probabilities_dict())
    qc.append(V_gate.inverse(),qargs=list(np.arange(0,number_ancillas))[::1])
    print(Statevector(qc))
    print(Statevector(qc).probabilities_dict())
    qc.measure(list(np.arange(0,number_ancillas)),list(np.arange(0,number_ancillas)))
    qc.draw(output="mpl")
    plt.show()
    job = execute(qc, backend, shots=10000)
    result = job.result()
    outputstate = result.get_statevector(qc)
    print(result.get_counts())
    c=np.sqrt(result.get_counts()["0"])
    print("c: %f"%c)
    print(outputstate)
    print(Statevector(outputstate).probabilities_dict())#.draw(output="repr")
def LCU_2(left_unitaries,right_unitaries,left_parameters,right_parameters,num_qubits,backend=None):
    number_ancillas=int(np.ceil(np.log2(len(left_unitaries))))
    number_circuit_qubits=number_ancillas+num_qubits+1
    qc=QuantumCircuit()
    qr=QuantumRegister(number_circuit_qubits,"q")
    cr=ClassicalRegister(number_ancillas*2,"c")
    qc.add_register(qr)
    qc.add_register(cr)
    controlled_unitaries_left=[]
    controlled_unitaries_right=[]
    V_L0=np.pad(np.sqrt(left_parameters),(0,int(2**number_ancillas-len(left_parameters))))
    V_L0=V_L0/np.linalg.norm(V_L0)
    c1=np.sum(V_L0**2)
    print(c1)
    V_R0=np.pad(np.sqrt(right_parameters),(0,int(2**number_ancillas-len(right_parameters))))
    V_R0=V_R0/np.linalg.norm(V_R0)
    c2=np.sum(V_R0**2)
    print(c2)
    XL=np.random.rand(len(V_L0),len(V_L0))
    XL[:,0]=V_L0
    V_L,R=np.linalg.qr(XL)
    if R[0,0]<0:
        V_L[:,0]=-V_L[:,0]
    XR=np.random.rand(len(V_R0),len(V_R0))
    XR[:,0]=V_R0
    V_R,R=np.linalg.qr(XR)
    if R[0,0]<0:
        V_R[:,0]=-V_R[:,0]
    V_L_gate=UnitaryGate(Operator(V_L))
    V_R_gate=UnitaryGate(Operator(V_R))
    controlled_VL=V_L_gate.control(num_ctrl_qubits=1, label=None, ctrl_state="0")
    controlled_VR=V_R_gate.control(num_ctrl_qubits=1, label=None, ctrl_state="1")
    for k,unitary in enumerate(left_unitaries):
        controlled_unitaries_left.append(unitary.control(number_ancillas+1,ctrl_state=get_bin(k,number_ancillas)+"0"))
    for k in range(len(right_unitaries)):
        controlled_unitaries_right.append(right_unitaries[k].control(number_ancillas+1,ctrl_state=get_bin(k,number_ancillas)+"1"))
    qc.h(0)
    qc.append(controlled_VL,qargs=list(np.arange(0,number_ancillas+1)))

    for k,uni in enumerate(controlled_unitaries_left):
        qc.append(uni,qr)

    qc.append(controlled_VL.inverse(),qargs=list(np.arange(0,number_ancillas+1)))
    qc.measure(list(np.arange(1,number_ancillas+1)),list(np.arange(0,number_ancillas)))
    qc.append(controlled_VR,qargs=list(np.arange(0,number_ancillas+1)))
    for k,unitary in enumerate(controlled_unitaries_right):
        qc.append(unitary,qr)

    qc.append(controlled_VR.inverse(),qargs=list(np.arange(0,number_ancillas+1)))

    qc.measure(list(np.arange(1,number_ancillas+1)),list(np.arange(number_ancillas,2*number_ancillas)))
    qc.decompose().draw(output="mpl")
    #plt.show()
    shots=10000
    job = execute(qc, backend, shots=shots)
    result = job.result()
    print(result.get_counts())
    p1=0
    for i in range(2):
        try:
            p1+=result.get_counts()["%d0"%i]/shots
        except KeyError:
            pass
    p2=0
    for i in range(2):
        try:
            p2+=result.get_counts()["0%d"%i]/shots/p1
        except KeyError:
            pass
    outputstate = result.get_statevector(qc)
    overlap_exp=(I^I^X)#+1j*(I^I^Y)
    matrix=Operator(overlap_exp).data
    vec=Statevector(outputstate).data
    print("Overlap is found to be")
    c1=1/np.sqrt(2*(p1-1/2))
    c2=1/np.sqrt(2*(p2*p1-1/(2*c1**2)))
    print(1/(2*c1**2*p1*p2),1/(2*c2**2*p1*p2))
    print((c1**2+c2**2)/(2*c1*c2)*vec.T@matrix@vec)
    print(p1*p2*(c1*c2)*vec.T@matrix@vec)
    print(Statevector(outputstate))
    print(Statevector(outputstate).probabilities_dict())#.draw(output="repr")
def get_01_state(unitary1,unitary2,num_qubits,backend=None):
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
    if backend is None:
        newcirc=qc
    else:
        newcirc=transpile(qc,backend=backend,optimization_level=1)
    return newcirc
def get_energy_expectations(zero1_state,op_list,qi):
    print("Starting shit")
    expectation_vals={}
    state=CircuitStateFn(zero1_state)

    number_OPs=len(op_list)
    dicterino={}
    maxnum=30
    k=0
    while k<number_OPs:
        print(k)
        op_list_new=[]
        for op in op_list[k:k+maxnum]:
            op_list_new.append(op^X)
        measurable_expression = StateFn(ListOp(op_list_new), is_measurement=True).compose(state)
        expectation = AerPauliExpectation().convert(measurable_expression)
        sampler = CircuitSampler(qi).convert(expectation)
        values=sampler.eval()
        dicterino.update({str(op_list[k+i]): values[i] for i in range(len(values))})
        print(values)
        k+=maxnum
    return dicterino
def get_overlap_expectations(zero1_state,num_qubits,qi):
    state=CircuitStateFn(zero1_state)
    overlap_measurement=I
    for i in range(num_qubits-1):
        overlap_measurement=overlap_measurement^I
    overlap_measurement=overlap_measurement^X
    measurable_expression = StateFn(overlap_measurement, is_measurement=True).compose(state)
    expectation = AerPauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(qi).convert(expectation)
    return sampler.eval()
def get_energy_expectations_01state(zero1_state,num_qubits,qubit_hamiltonian,qi):
    state=CircuitStateFn(zero1_state)
    energy_measurement=qubit_hamiltonian^X
    measurable_expression = StateFn(energy_measurement, is_measurement=True).compose(state)
    expectation = AerPauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(qi).convert(expectation)
    return sampler.eval()
def calculate_energy_overlap(unitary1,unitary2,num_qubits,hamiltonian,qi,nuc_rep,include_custom=True,complex=False):
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
    return sampler_energy.eval()+sampler_overlap.eval()*nuc_rep,sampler_overlap.eval()

def get_tapering_value(index,max_index):
    bitstring=bin(index)[2:]
    maxlen_bitstring=bin(max_index-1)[2:]
    z2_symmetries=[1]*(len(maxlen_bitstring)-len(bitstring))
    for bit in bitstring:
        if bit=="1":
            z2_symmetries.append(-1)
        else:
            z2_symmetries.append(1)
    return z2_symmetries
def find_symmetry(molecule,x,qi,qubit_converter_nosymmetry,active_space,nelec,basis):
    hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,x,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=None)
    qubit_op = qubit_converter_nosymmetry.convert(hamiltonian,num_particles=num_particles)
    pauli_symm = Z2Symmetries.find_Z2_symmetries(qubit_op)
    qubit_op_new=pauli_symm.taper(qubit_op).reduce()
    vals=np.zeros(len(qubit_op_new))
    for i in range(len(qubit_op_new)):
        result = NumPyEigensolver().compute_eigenvalues(qubit_op_new[i])
        print("Eigenvalue of %f"%np.real(result.eigenvalues[0]))
        #print(pauli_symm.tapering_values)
        vals[i]=np.real(result.eigenvalues[0])
    minimal=int(np.argmin(vals))
    tapering_value=get_tapering_value(minimal,len(qubit_op_new))
    return tapering_value
    return unitary,energy,optimal_params
def get_transformation_circuit(R,active_space,nelec):
    n_qubits=len(active_space)
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit(openfermion.bogoliubov_transform(qubits,R))
    circuit_qasm=circuit.to_qasm()
    outfile=open("qasm_temp.txt","w")
    outfile.write(circuit_qasm)
    outfile.close()
    circuit_qi=QuantumCircuit.from_qasm_file("qasm_temp.txt")
    return circuit_to_gate(transpile(circuit_qi,optimization_level=3))

if __name__=="__main__":
    basis="STO-6G"
    ref_x=2
    mol = mol = gto.M(atom=molecule(ref_x), basis=basis,unit="Bohr")
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    nuc_rep=mf.energy_nuc()
    ref_det=mf.mo_coeff
    active_space=[1,2,3,5,6] #Freezing [1,2,3,5,6] works kinda for BeH2
    nelec=4
    sample_x=[2,3,4,5,6]
    x_of_interest=np.linspace(2,7,51)
    E_EVC=np.zeros(len(x_of_interest))
    E_exact=np.zeros(len(x_of_interest))
    E_UCC=np.zeros(len(x_of_interest))
    E_k2=np.zeros(len(x_of_interest))
    #backend=Aer.get_backend("aer_simulator_statevector")
    backend= Aer.get_backend('statevector_simulator')
    seed=np.random.randint(100000)
    qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
    #optimizer=SLSQP(maxiter=500)
    #optimizer=SPSA(maxiter=100)
    optimizer=SciPyOptimizer("BFGS")#L_BFGS_B()
    numPyEigensolver=NumPyEigensolver()

    qubit_converter_nosymmetry = QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True)
    tapering_value=find_symmetry(molecule,ref_x,qi,qubit_converter_nosymmetry,active_space,nelec,basis)
    qubit_converter_symmetry=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True,z2symmetry_reduction=tapering_value)
    #qubit_converter_symmetry=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True)

    unitaries=[]
    optimal_params=None
    optimal_params_k2=None
    UCC_1_params=[]
    UCC_2_params=[]
    """
    for i,x in enumerate(sample_x):
        hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,x,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
        qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
        num_qubits=qubit_hamiltonian.num_qubits
        print("Num qubits: %d"%num_qubits)
        result = numPyEigensolver.compute_eigenvalues(qubit_hamiltonian)
        print("Exact: %f"%np.real(result.eigenvalues[0]+nuc_rep))
        ansatz,initpoint=kUpUCCSD_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=1)
        if optimal_params is None:
            optimal_params=initpoint
        #ansatz,initial_point=UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=1,generalized=True)

        #unitary,energy,optimal_params=get_unitary(qubit_hamiltonian,ansatz,optimizer,qi,include_custom=True,initial_point=optimal_params,nuc_rep=nuc_rep)
        #unitary,energy,optimal_params=optimize_stepwise(num_particles,num_spin_orbitals,num_qubits,qubit_converter_symmetry,qubit_hamiltonian,kUpUCCSD_ansatz,ansatz,optimizer,qi,include_custom=True,k=3,initial_point=None)

        #print("Approximate 3-fold: %f"%energy)
        ansatz,initial_point_UCC=UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=1,generalized=True)
        if optimal_params_UCC is None:
            optimal_params_UCC=initpoint_UCC
        unitary,energy,optimal_params_UCC=get_unitary(qubit_hamiltonian,ansatz,optimizer,qi,include_custom=True,initial_point=optimal_params_UCC,nuc_rep=nuc_rep)
        unitaries.append(unitary)
        print("Approximate UCC: %f"%energy)
    """
    for k,x in enumerate(x_of_interest):

        H=np.zeros((len(sample_x),len(sample_x)))
        S=np.zeros((len(sample_x),len(sample_x)))
        hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,x,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
        qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
        num_qubits=qubit_hamiltonian.num_qubits
        """
        for i in range(len(unitaries)):
            for j in range(i,len(unitaries)):
                h,s=calculate_energy_overlap(unitaries[i],unitaries[j],num_qubits,qubit_hamiltonian,qi,nuc_rep,include_custom=True,complex=False)
                S[i,j]=S[j,i]=s
                H[i,j]=H[j,i]=h
        e,cl,c=eig(scipy.linalg.pinv(S,atol=1e-8)@H,left=True)
        idx = np.real(e).argsort()
        e = e[idx]
        c = c[:,idx]
        cl = cl[:,idx]
        E_EVC[k]=np.real(e[0])
        """
        result = numPyEigensolver.compute_eigenvalues(qubit_hamiltonian)
        E_exact[k]=np.real(result.eigenvalues[0]+nuc_rep)
        ansatz_k2,initial_point_k2=UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=2,generalized=False)
        if optimal_params_k2 is None:
            optimal_params_k2=initial_point_k2
        unitary,energy_k2,optimal_params_k2=get_unitary(qubit_hamiltonian,ansatz_k2,optimizer,qi,include_custom=True,initial_point=optimal_params_k2,nuc_rep=nuc_rep)
        E_k2[k]=np.real(energy_k2)
        ansatz,initial_point=UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=1,generalized=False)
        if optimal_params is None:
            optimal_params=initial_point
        unitary,energy,optimal_params=get_unitary(qubit_hamiltonian,ansatz,optimizer,qi,include_custom=True,initial_point=optimal_params,nuc_rep=nuc_rep)
        E_UCC[k]=np.real(energy)
        print("Exact: %f, UCC: %f, k2: %f"%(E_exact[k],E_UCC[k],E_k2[k]))
    plt.plot(x_of_interest,E_exact,"-",label="exact diagonalization")
    plt.plot(x_of_interest,E_UCC,"-",label="UCC")
    plt.plot(x_of_interest,E_k2,"-",label="double UCC")
    #plt.plot(x_of_interest,E_EVC,"-",label="EVC")
    plt.legend()
    plt.show()
