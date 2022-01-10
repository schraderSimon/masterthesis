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
from qiskit.algorithms.optimizers import COBYLA, SPSA,SLSQP, QNSPSA
from qiskit_nature.circuit.library import UCC,UCCSD, HartreeFock
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
import sys
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
def orthogonal_procrustes(mo_new,reference_mo):
    A=reference_mo.T
    B=mo_new.T
    M=B@A.T
    U,s,Vt=scipy.linalg.svd(M)
    return U@Vt, 0

def localize_procrustes(mol,mo_coeff,mo_occ,ref_mo_coeff,mix_states=False,active_orbitals=None,nelec=None):
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

    if mix_states==False:
        mo=mo_coeff[:,active_orbitals_occ]
        premo=ref_mo_coeff[:,active_orbitals_occ]
        R,scale=orthogonal_procrustes(mo,premo)
        mo=mo@R

        mo_coeff[:,active_orbitals_occ]=np.array(mo)
        mo_unocc=mo_coeff[:,active_orbitals_unocc]
        premo=ref_mo_coeff[:,active_orbitals_unocc]
        R,scale=orthogonal_procrustes(mo_unocc,premo)
        mo_unocc=mo_unocc@R

        mo_coeff[:,active_orbitals_unocc]=np.array(mo_unocc)


    elif mix_states==True:
        mo=mo_coeff[:,active_orbitals]
        premo=ref_mo_coeff[:,active_orbitals]
        R,scale=orthogonal_procrustes(mo,premo)
        mo=mo@R

        mo_coeff[:,active_orbitals]=np.array(mo)
    return mo_coeff

def molecule(x):
    return "Li 0 0 0; H 0 0 -%f"%(x)
#molecule=lambda x: "H 0 0 0; H 0 0 -%f"%x
def get_qubit_op(molecule,x,qi,qubit_converter,basis="STO-6G",ref_det=None,remove_core=True,active_space=None,nelec=None):

    mol = mol = gto.M(atom=molecule(x), basis=basis,unit="Bohr")
    mol.build()
    mf = scf.RHF(mol)
    eref=mf.kernel()
    hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    nuc_rep=mf.energy_nuc()
    mo_coeff=mf.mo_coeff
    print(mo_coeff)
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
    qubit_op = qubit_converter.convert(hamiltonian,num_particles=num_particles)
    return qubit_op, num_particles,num_spin_orbitals, nuc_rep+energyshift
def get_unitary(hamiltonian,ansatz,optimizer,qi,include_custom=True):
    vqe = VQE(ansatz=ansatz,include_custom=include_custom, optimizer=optimizer,quantum_instance=qi)
    vqe_result =vqe.compute_minimum_eigenvalue(hamiltonian)
    circuit=vqe.get_optimal_circuit()
    unitary=circuit_to_gate(circuit)
    return unitary, vqe_result.eigenvalue.real+nuc_rep
def SU2_ansatz(num_particles,num_spin_orbitals,qubit_converter=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True),reps=1):
    pass
def get_ansatz(num_particles,num_spin_orbitals,qubit_converter=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True),reps=1):
    initial_state = HartreeFock(
            num_spin_orbitals=num_spin_orbitals,
            num_particles=num_particles,
            qubit_converter=qubit_converter,
        )
    reps=reps
    var_form = UCC(
        excitations="sd",
        num_particles=num_particles,
        num_spin_orbitals=num_spin_orbitals,
        initial_state=initial_state,
        qubit_converter=qubit_converter,
        reps=reps
    )
    return var_form
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
basis="STO-6G"
ref_x=3
mol = mol = gto.M(atom=molecule(ref_x), basis=basis,unit="Bohr")
mol.build()
mf = scf.RHF(mol)
mf.kernel()
hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
nuc_rep=mf.energy_nuc()
ref_det=mf.mo_coeff

sample_x=[2,3,4,5,6]
x_of_interest=np.linspace(1,6,26)
E_EVC=np.zeros(len(x_of_interest))
E_exact=np.zeros(len(x_of_interest))
E_UCC=np.zeros(len(x_of_interest))
backend=Aer.get_backend("aer_simulator_statevector")
seed=100
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
qubit_converter = QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True)#,z2symmetry_reduction = 'auto')
numPyEigensolver=NumPyEigensolver()
optimizer=SLSQP(maxiter=200)
active_space=[1,2,4]
nelec=2
unitaries=[]
for i,x in enumerate(sample_x):
    qubit_hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_qubit_op(molecule,x,qi,qubit_converter,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
    ansatz=get_ansatz(num_particles,num_spin_orbitals,qubit_converter=qubit_converter)
    #fidelity = QNSPSA.get_fidelity(ansatz)
    #optimizer = QNSPSA(fidelity, maxiter=500)
    result = numPyEigensolver.compute_eigenvalues(qubit_hamiltonian)
    print("Exact: %f"%np.real(result.eigenvalues[0]+nuc_rep))
    qubit_hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_qubit_op(molecule,x,qi,qubit_converter,active_space=active_space,nelec=nelec,basis=basis,ref_det=None)
    ansatz=get_ansatz(num_particles,num_spin_orbitals,qubit_converter=qubit_converter)
    #fidelity = QNSPSA.get_fidelity(ansatz)
    #optimizer = QNSPSA(fidelity, maxiter=500)
    #result2 = numPyEigensolver.compute_eigenvalues(qubit_hamiltonian)
    #print("Exact 2-exact: %e"%np.real(result2.eigenvalues[0]-result.eigenvalues[0]))
    unitary,energy=get_unitary(qubit_hamiltonian,ansatz,optimizer,qi,include_custom=True)
    unitaries.append(unitary)
    num_qubits=unitary.num_qubits
    print("Approx: %f"%energy)
for k,x in enumerate(x_of_interest):
    H=np.zeros((len(sample_x),len(sample_x)))
    S=np.zeros((len(sample_x),len(sample_x)))
    qubit_hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_qubit_op(molecule,x,qi,qubit_converter,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
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
    result = numPyEigensolver.compute_eigenvalues(qubit_hamiltonian)
    E_exact[k]=np.real(result.eigenvalues[0]+nuc_rep)
    unitary,energy=get_unitary(qubit_hamiltonian,ansatz,optimizer,qi,include_custom=True)
    #E_UCC[k]=np.real(energy)
    print(E_UCC[k],E_exact[k],E_EVC[k])
plt.plot(x_of_interest,E_exact,"-",label="exact diagonalization")
plt.plot(x_of_interest,E_UCC,"-",label="UCC")
plt.plot(x_of_interest,E_EVC,"-",label="EVC")
plt.legend()
plt.show()
