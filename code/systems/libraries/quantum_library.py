from func_lib import *

from numba import jit
from matrix_operations import *
from helper_functions import *
import openfermion, cirq
from qiskit_nature.circuit.library import UCC,UCCSD, HartreeFock, PUCCD, SUCCD
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy, ParticleNumber
from qiskit_nature.properties.second_quantization.electronic.electronic_structure_driver_result import ElectronicStructureDriverResult
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis, ElectronicBasisTransform
from qiskit_nature.algorithms import VQEUCCFactory
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper,BravyiKitaevMapper, BravyiKitaevSuperFastMapper
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    ElectronicIntegrals,
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
    IntegralProperty,
)
from qiskit_nature.properties import GroupedProperty
from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer, FreezeCoreTransformer

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, transpile, execute

from qiskit.opflow.list_ops import ListOp
from qiskit.opflow.primitive_ops import Z2Symmetries, PauliOp
from qiskit.opflow import X, Y, Z, I
from qiskit.opflow import CircuitStateFn, TensoredOp,PauliExpectation, CircuitSampler, StateFn, AerPauliExpectation, MatrixExpectation

from qiskit.extensions import UnitaryGate, Initialize
from qiskit.providers.aer import AerSimulator, QasmSimulator
from qiskit.algorithms.optimizers import *
from qiskit.algorithms import NumPyEigensolver,VQE
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import EfficientSU2, QAOAAnsatz, StatePreparation, ExcitationPreserving, RealAmplitudes
from qiskit.converters import circuit_to_gate
from qiskit.tools.visualization import circuit_drawer

from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Operator, Pauli


warnings.filterwarnings('ignore', category=DeprecationWarning)


def get_basis_Hamiltonian(molecule,x,qi,mo_coeff,basis="STO-6G",active_space=None,nelec=None,symmetry=None,irreps=None):
    """
    Returns the second quantized representation of the Hamiltonian. Does not use procrustes orbitals.

    Input:
    molecule (function): Function to obtain molecule
    x: The position
    qi: Quantum Instance (not neccessary)
    mo_coeff: Coeffient matrix
    basis: basis
    active_space (list): active space
    nelec: Number of electrons in active space
    symmetry, irreps: Symmetry and irreps

    Returns:
    Hamiltonian : second quantized hamiltonian
    num_particles: Number of electrons
    num_spin_orbitals: Number of spin orbitals
    nuc_rep+energyshift: Constant contribution to the energy
    newGroup: Electronc structure driver result for the molecule
    """
    mol = mol = gto.M(atom=molecule(x), basis=basis,unit="Bohr")#,symmetry=symmetry)
    mol.build()
    mf = scf.RHF(mol)
    #mf.irrep_nelec=irreps
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
    """
    Returns the second quantized representation of the Hamiltonian. Uses procrustes orbitals!!

    Input:
    molecule (function): Function to obtain molecule
    x: The position
    qi: Quantum Instance (not neccessary)
    ref_det: Coeffient matrix to similarize to
    basis: basis
    active_space (list): active space
    nelec: Number of electrons in active space
    symmetry, irreps: Symmetry and irreps

    Returns:
    Hamiltonian : second quantized hamiltonian
    num_particles: Number of electrons
    num_spin_orbitals: Number of spin orbitals
    nuc_rep+energyshift: Constant contribution to the energy
    newGroup: Electronc structure driver result for the molecule
    """
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
    """
    Returns the unitary corresponding to the optimized ansatz.

    Input:
    Hamiltonian: The Hamiltonian in qubit representation
    Ansatz: The ansatz (e.g. UCCSD, HEA)
    optimizer: The classical optimization algorithm
    qi: Quantum instance
    nuc_rep: Constant part of the energy
    include_custom: Sampled vs. calculated expectations
    initial point: Initial set of parameters

    Returns:
    unitary corresponding to the optimized ansatz, the energy, and the optimal parameters
    """
    vqe = VQE(ansatz=ansatz,include_custom=include_custom, optimizer=optimizer,quantum_instance=qi,initial_point=initial_point)

    vqe_result =vqe.compute_minimum_eigenvalue(hamiltonian)
    #circuit=vqe_result.ansatz.bind_parameters(vqe_result.optimal_point)
    circuit=None
    unitary=None#circuit_to_gate(circuit)
    optimal_params=vqe_result.optimal_point
    return unitary, vqe_result.eigenvalue.real+nuc_rep, optimal_params
def k_fold_unitary(num_particles,num_spin_orbitals,num_qubits,qubit_converter,hamiltonian,ansatz_func,ansatz_0,optimizer,qi,include_custom=True,k=3,initial_point=None):
    """
    Returns the unitary corresponding to the optimized ansatz after repeated applications.

    Input:
    Hamiltonian: The Hamiltonian in qubit representation
    Ansatz: The ansatz (e.g. UCCSD, HEA)
    optimizer: The classical optimization algorithm
    qi: Quantum instance
    nuc_rep: Constant part of the energy
    include_custom: Sampled vs. calculated expectations
    initial point: Initial set of parameters

    Returns:
    unitary corresponding to the optimized ansatz, the energy, and the optimal parameters
    """
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

def UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True),reps=1,initial_state=None,generalized=False):
    "Returns UCCSD ansatz circuit + initial guess vector"
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
    var_form.excitation_ops()
    init_pt=(np.random.rand(len(var_form.parameters))-0.5)

    print(len(init_pt))
    return var_form, init_pt
def get_01_state(unitary1,unitary2,num_qubits,backend=None):
    """
    Given two unitaries, implements the state $\Omega$.

    Returns: The circuit to implement the state $\Omega$.
    """
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
        newcirc=transpile(qc,backend=backend,optimization_level=1) #Optimization
    return newcirc
def get_energy_expectations(zero1_state,op_list,qi):
    """
    Given the $\Omega$ state, calculates the expectation values for a list of operators. Assumes REAL expectation values.

    Returns:
    A dictionary matching the op strings to expectation values.
    """
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
    """
    Given the $\Omega$ state, calculates the overlap between $\Psi_1$ and $Psi_2$.
    It assumes that the imaginary part is zero, e.g. that the wave functions are real!
    Returns:
    The overlap.
    """
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
    """
    Given the $\Omega$ state, calculates the Hamiltonian element between $\Psi_1$ and $Psi_2$.
    It assumes that the imaginary part is zero, e.g. that the wave functions are real!
    Returns:
    The overlap.
    """
    state=CircuitStateFn(zero1_state)
    energy_measurement=qubit_hamiltonian^X
    measurable_expression = StateFn(energy_measurement, is_measurement=True).compose(state)
    expectation = AerPauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(qi).convert(expectation)
    return sampler.eval()

def _get_tapering_value(index,max_index):
    """Convenience function to obtain qiskit-readable list"""
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
    """
    find the tapering value, e.g. the eigenvalues of the to-be-tapered-off qubits (e.g. finds the correct symmetry sector)
    """
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
def get_transformation_circuit(R,active_space,nelec):
    """
    Given a unitary rotation R=e^(-\kappa), returns a gate that implements it.
    """
    n_qubits=len(active_space)
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit(openfermion.bogoliubov_transform(qubits,R))
    circuit_qasm=circuit.to_qasm()
    outfile=open("qasm_temp.txt","w")
    outfile.write(circuit_qasm)
    outfile.close()
    circuit_qi=QuantumCircuit.from_qasm_file("qasm_temp.txt")
    return circuit_to_gate(transpile(circuit_qi,optimization_level=3))
