from pyscf import gto, scf, symm, mcscf, fci, cc, mp
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy
import matplotlib.pyplot as plt
import numpy as np
import openfermion, cirq
from qiskit import QuantumCircuit, transpile
from measure_overlap import *
np.set_printoptions(linewidth=300,precision=10,suppress=True)

def get_geometry(x):
    y = lambda x: 2.54 - 0.46*x
    atom="H  " + str(-y(x)) + " 0 " + str(x) + "; H " + str(y(x)) + " 0  " + str(x) + "; Be 0 0 0"
    return atom
def get_geometry2(x):
    return "H 0 0 %f; H 0 0 -%f; Be 0 0 0"%(x,x)
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

basis="STO-6G"
occdict1={"A1":6,"B1":0,"B2":0}
occdict2={"A1":4,"B1":2,"B2":0}
occdict3={"A1":4,"B1":0,"B2":2}

xs=np.linspace(0,4,5)
occdicts=[occdict1,occdict2,occdict3]
energies=np.zeros((len(xs),3))
energies_min=np.zeros(len(xs))
mo_coeff_min=[]
molecular_orbital_energies=[]
mo_coeff_1=[]
enuc=[]
irreps=[]
for k,x in enumerate(xs):
    print(x)
    atom=get_geometry(x)
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
    mo_coeff_temp=[]
    mo_en_temp=[]
    for i in [0,1,2]:
        mf = scf.RHF(mol)
        mf.verbose=0
        mf.irrep_nelec=occdicts[i]
        e=mf.kernel(verbose=0)
        mo_coeff_temp.append(mf.mo_coeff)
        mo_en_temp.append(mf.mo_energy)
        energies[k,i]=e
    emindex=np.argmin(energies[k,:])
    irreps.append(occdicts[emindex])
    mo_coeff_min.append(mo_coeff_temp[emindex])
    mo_coeff_1.append(mf)
    enuc.append(mf.energy_nuc())
    molecular_orbital_energies.append(mo_en_temp[2])
    energies_min[k]=energies[k,emindex]
mo_coeff_min=[]
for k,x in enumerate(xs):
    print(x)
    atom=get_geometry(x)
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
    mf = scf.RHF(mol)
    e=mf.kernel(verbose=0)
    mo_coeff_min.append(mf.mo_coeff)

"""
plt.plot(xs,energies[:,0],label="irrep1")
plt.plot(xs,energies[:,1],label="irrep2")
plt.plot(xs,energies[:,2],label="irrep3")
plt.plot(xs,energies_min,label="minimal")
E_casci=[]
for mf in mo_coeff_1:
    mc = mcscf.CASSCF(mf, 3, 4)
    E_casci.append(mc.kernel()[0])
plt.plot(xs,E_casci,label="CASCI")
plt.legend()
plt.show()
"""
#plt.plot(xs,molecular_orbital_energies)
#plt.show()
"""
ref_det=mo_ref=mo_coeff_min[0]
mo_occ=mf.mo_occ
norm_diffs_1=np.zeros(len(xs))
norm_diffs_2=np.zeros(len(xs))
norm_diffs_3=np.zeros(len(xs))
minMos=[]
active_orbitals=[1,2,3,4,5,6]
for i in range(len(xs)):
    print(xs[i])
    mo_coeff_new1=localize_procrustes(mol,mo_coeff_min[i],mo_occ,mo_ref,mix_states=True,active_orbitals=active_orbitals,nelec=4)
    mo_coeff_new2=localize_procrustes(mol,mo_coeff_min[i],mo_occ,mo_ref,mix_states=True,active_orbitals=None,nelec=None)

    norm_diffs_1[i]=np.linalg.norm(mo_coeff_new1-mo_ref,"fro")
    norm_diffs_2[i]=np.linalg.norm(mo_coeff_new2-mo_ref,"fro")
#plt.plot(0.52918*xs,norm_diffs_1,label="firsto")
#plt.plot(0.52918*xs,norm_diffs_2,label="secondo")
#plt.legend()
#plt.show()

"""

backend= Aer.get_backend('statevector_simulator')
seed=np.random.randint(100000)
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
optimizer=SciPyOptimizer("BFGS")
numPyEigensolver=NumPyEigensolver()

qubit_converter = QubitConverter(mapper=JordanWignerMapper())
active_space=[0,1,2,3,4,5,6]
nelec=6
energies=[]
energies_transformed=[]
HF_energies=[]
HF_energies_transformed=[]
for k,x in enumerate(xs):
    if k==0:
        continue
    qc=QuantumCircuit(len(active_space)*2)
    for i in range(2):
        for j in range(nelec//2):
            qc.x(j+i*len(active_space))
    #sys.exit(1)
    hamiltonian,num_particles,num_spin_orbitals,nuc_rep,orig_group=get_basis_Hamiltonian(get_geometry,x,qi,mo_coeff_min[k],basis="STO-6G",active_space=active_space,nelec=nelec,symmetry="C2v")#,irreps=irreps[k])
    qubit_hamiltonian=qubit_converter.convert(hamiltonian)
    #ansatz,init_point=UCC_ansatz(num_particles,num_spin_orbitals,n_qubits,qubit_converter=qubit_converter,reps=1,initial_state=None,generalized=False)
    ansatz=HartreeFock(num_spin_orbitals=num_spin_orbitals,num_particles=num_particles,qubit_converter=qubit_converter)
    state=CircuitStateFn(ansatz)
    measurable_expression = StateFn(qubit_hamiltonian, is_measurement=True).compose(state)
    expectation = AerPauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(qi).convert(expectation)
    value=sampler.eval()
    print(value+nuc_rep)
    #result = NumPyEigensolver().compute_eigenvalues(qubit_hamiltonian)
    #energies.append(np.real(result.eigenvalues[0])+nuc_rep)
    HF_energies.append(value+nuc_rep)
    procrustes_state,R=localize_procrustes(mol,mo_coeff_min[k],mf.mo_occ,mo_coeff_min[0],mix_states=True,active_orbitals=active_space,nelec=nelec,return_R=True)
    hamiltonian1,num_particles,num_spin_orbitals,nuc_rep,ref_group=get_basis_Hamiltonian(get_geometry,x,qi,procrustes_state,basis="STO-6G",active_space=active_space,nelec=nelec,symmetry="C2v")#,irreps=irreps[k])
    #qubit_hamiltonian = qubit_converter.convert(hamiltonian1)
    ansatz=HartreeFock(num_spin_orbitals=num_spin_orbitals,num_particles=num_particles,qubit_converter=qubit_converter)
    gate=get_transformation_circuit(R,active_space,nelec)
    n_qubits=len(active_space)
    ansatz.append(gate,list(np.arange(n_qubits-1,-1,-1)))
    ansatz.append(gate,list(np.arange(n_qubits-1,-1,-1)+n_qubits))
    state=CircuitStateFn(ansatz)
    measurable_expression = StateFn(qubit_hamiltonian, is_measurement=True).compose(state)
    expectation = AerPauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(qi).convert(expectation)
    value=sampler.eval()
    print(value+nuc_rep)
    sys.exit(1)
    HF_energies_transformed.append(value+nuc_rep)
    #result1 = NumPyEigensolver().compute_eigenvalues(qubit_hamiltonian)
    #energies_transformed.append(np.real(result1.eigenvalues[0])+nuc_rep)
plt.plot(xs,energies,label="energies")
plt.plot(xs,HF_energies,label="HF")
plt.plot(xs,energies_transformed,label="transformed")
plt.plot(xs,HF_energies_transformed,label="HF transformed")

plt.legend()
plt.show()
