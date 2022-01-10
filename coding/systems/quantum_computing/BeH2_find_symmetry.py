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
from qiskit.opflow.primitive_ops import Z2Symmetries

import sys
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
def molecule(x):
    return "Li 0 0 0; H 0 0 -%f"%(x)#; H 0 0 %f"%(x,x)
def get_qubit_op(molecule,x,qi,qubit_converter,basis="STO-6G",ref_det=None,remove_core=True,active_space=None,nelec=None):
    mol = mol = gto.M(atom=molecule(x), basis=basis,unit="Bohr",symmetry=True)
    mol.build()
    mf = scf.RHF(mol)
    eref=mf.kernel()
    hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    nuc_rep=mf.energy_nuc()
    mo_coeff=mf.mo_coeff
    print(mo_coeff)
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
    particle_number = ParticleNumber(num_spin_orbitals=num_spin_orbitals,num_particles=num_particles)
    basistransform=ElectronicBasisTransform(ElectronicBasis.AO,ElectronicBasis.MO,mo_coeff)
    grouped_property=ElectronicStructureDriverResult()
    grouped_property.add_property(electronic_energy)
    grouped_property.add_property(particle_number)
    grouped_property.add_property(basistransform)
    transformer= ActiveSpaceTransformer(nelec, len(active_space), active_space)
    newGroup=transformer.transform(grouped_property)
    return newGroup

basis="STO-6G"
ref_x=3
mol = mol = gto.M(atom=molecule(ref_x), basis=basis,unit="Bohr")
mol.build()
mf = scf.RHF(mol)
mf.kernel()
hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
nuc_rep=mf.energy_nuc()
ref_det=mf.mo_coeff

sample_x=np.linspace(0.5,5,25)
x_of_interest=np.linspace(1,6,26)
E_EVC=np.zeros(len(x_of_interest))
E_exact=np.zeros(len(x_of_interest))
E_UCC=np.zeros(len(x_of_interest))
backend=Aer.get_backend("aer_simulator_statevector")
seed=100
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
qubit_converter = QubitConverter(mapper=JordanWignerMapper(),two_qubit_reduction=False)
numPyEigensolver=NumPyEigensolver()
optimizer=COBYLA(maxiter=200)
active_space=[1,2,3,4]
nelec=2
unitaries=[]
def get_tapering_value(index,max_index):
    print(index,max_index)
    bitstring=bin(index)[2:]
    maxlen_bitstring=bin(max_index-1)[2:]
    z2_symmetries=[1]*(len(maxlen_bitstring)-len(bitstring))
    for bit in bitstring:
        if bit=="1":
            z2_symmetries.append(-1)
        else:
            z2_symmetries.append(1)
    print(z2_symmetries)
    return z2_symmetries
def find_symmetry():
    newGroup=get_qubit_op(molecule,1.5,qi,qubit_converter,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
    energyshift=np.real(newGroup.get_property("ElectronicEnergy")._shift["ActiveSpaceTransformer"])
    nuc_rep=np.real(newGroup.get_property("ElectronicEnergy")._nuclear_repulsion_energy)
    hamiltonian = newGroup.second_q_ops()[0]
    num_particles=newGroup.get_property("ParticleNumber").num_particles
    num_spin_orbitals=newGroup.get_property("ParticleNumber").num_spin_orbitals
    qubit_op = qubit_converter.convert(hamiltonian,num_particles=num_particles)
    pauli_symm = Z2Symmetries.find_Z2_symmetries(qubit_op)
    print(pauli_symm)
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
tapering_value=find_symmetry()
energies_exact=[]
energies_tapered=[]
energies_approx=[]
for i,x in enumerate(sample_x):

    newGroup=get_qubit_op(molecule,x,qi,qubit_converter,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
    energyshift=np.real(newGroup.get_property("ElectronicEnergy")._shift["ActiveSpaceTransformer"])
    nuc_rep=np.real(newGroup.get_property("ElectronicEnergy")._nuclear_repulsion_energy)
    hamiltonian = newGroup.second_q_ops()[0]
    num_particles=newGroup.get_property("ParticleNumber").num_particles
    num_spin_orbitals=newGroup.get_property("ParticleNumber").num_spin_orbitals
    qubit_op = qubit_converter.convert(hamiltonian,num_particles=num_particles)
    result0 = np.real(numPyEigensolver.compute_eigenvalues(qubit_op).eigenvalues[0])

    qubit_converter_symmetry=QubitConverter(mapper=JordanWignerMapper(),two_qubit_reduction=False,z2symmetry_reduction=tapering_value)
    qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian)
    result = np.real(numPyEigensolver.compute_eigenvalues(qubit_hamiltonian).eigenvalues[0])
    var_form = UCC(
        excitations="sd",
        num_particles=num_particles,
        num_spin_orbitals=num_spin_orbitals,
        #initial_state=initial_state,
        qubit_converter=qubit_converter_symmetry,
        reps=1
    )
    ansatz=var_form
    include_custom=True
    vqe = VQE(ansatz=ansatz,include_custom=include_custom, optimizer=optimizer,quantum_instance=qi)
    vqe_result =vqe.compute_minimum_eigenvalue(qubit_hamiltonian)
    energy=vqe_result.eigenvalue.real+nuc_rep+energyshift
    print(energy)
    print("Approx: %.15f"%energy)
    print("Exact: %.15f"%(nuc_rep+result0+energyshift))
    print("Exact: %.15f"%(nuc_rep+result+energyshift))
    energies_exact.append(nuc_rep+result0+energyshift)
    energies_tapered.append(nuc_rep+result+energyshift)
    energies_approx.append(energy)
plt.plot(sample_x,energies_exact,label="Exact untapered")
plt.plot(sample_x,energies_tapered,label="Exact tapered")
plt.plot(sample_x,energies_approx,label="Approx tapered")
plt.legend()
plt.show()
