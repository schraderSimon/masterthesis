from measure_overlap import *
from pyscf import gto, scf, ao2mo, fci, mcscf
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, cc,scf, ao2mo,fci
import openfermion, cirq

np.set_printoptions(linewidth=300,precision=10,suppress=True)
def molecule(x):
    y = lambda x: 2.54 - 0.46*x
    atom="H  " + str(-y(x)) + " 0 " + str(x) + "; H " + str(y(x)) + " 0  " + str(x) + "; Be 0 0 0"
    return atom

basis="STO-6G"
ref_x=1
mol = mol = gto.M(atom=molecule(ref_x), basis=basis,unit="Bohr")
mol.build()
mf = scf.RHF(mol)
mf.kernel()
hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
nuc_rep=mf.energy_nuc()
ref_det=mf.mo_coeff
active_space=[1,2,3,4,5,6] #Freezing [1,2,3,5,6] works kinda for BeH2
nelec=4
x_of_interest=np.linspace(0,4,41)
E_FCI=np.zeros(len(x_of_interest))
for k,x in enumerate(x_of_interest):
    atom=molecule(x)
    mol = gto.M(atom=atom, basis=basis, symmetry='C2v', unit='bohr')
    mf = scf.RHF(mol)
    e=mf.kernel(verbose=0)
    cisolver = fci.FCI(mf)
    E_FCI[k]=cisolver.kernel(verbose=0)[0]

E_EVC=np.zeros(len(x_of_interest))
E_exact=np.zeros(len(x_of_interest))
E_UCC=np.zeros(len(x_of_interest))
E_k2=np.zeros(len(x_of_interest))
backend= AerSimulator(method='statevector',max_parallel_threads=4)
seed=np.random.randint(100000)
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

optimizer=SciPyOptimizer("BFGS")#L_BFGS_B()
numPyEigensolver=NumPyEigensolver()

qubit_converter=QubitConverter(mapper=JordanWignerMapper())
UCC_UCC2_params=loadmat("SHIT_BeH2_UCC_vals.mat")

UCC_2_params=UCC_UCC2_params["UCC_1"]
xvals=UCC_UCC2_params["xvals"][0]
#sample_x=[0,1,2,3,4]
sample_x=[0,0.1,0.2,0.3,0.4]
occdict1={"A1":6,"B1":0,"B2":0}
occdict2={"A1":4,"B1":2,"B2":0}
occdict3={"A1":4,"B1":0,"B2":2}
occdicts=[occdict1,occdict2,occdict3]
energies=np.zeros((len(xvals),3))
irreps=[]
mo_coeff_min=[]
for k,x in enumerate(xvals):
    print(x)
    atom=molecule(x)
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



UCC_2_circuits=[]
hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,1,qi,active_space=active_space,nelec=nelec,basis=basis)
qubit_hamiltonian=qubit_converter.convert(hamiltonian,num_particles=num_particles)
#qubit_hamiltonian=qubit_hamiltonian.reduce()
print(qubit_hamiltonian)
num_qubits=qubit_hamiltonian.num_qubits


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
###
"""
sample_x_val=3
idx = (np.abs(xvals - sample_x_val)).argmin()
x=xvals[idx]
print(x)
UCC_2_param=UCC_2_params[idx]
UCC_2_ansatz_f,initial_point=UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter,reps=1,generalized=False)
UCC_2_ansatz=UCC_2_ansatz_f.assign_parameters(UCC_2_param)
mo_coeff=mo_coeff_min[idx]
hamiltonian, num_particles,num_spin_orbitals, nuc_rep,newGroup=get_basis_Hamiltonian(molecule,x,qi,mo_coeff,basis=basis,active_space=active_space,nelec=nelec,symmetry='C2v',irreps=irreps[k])
state=CircuitStateFn(UCC_2_ansatz)
qubit_hamiltonian=qubit_converter.convert(hamiltonian).reduce()
measurable_expression = StateFn(qubit_hamiltonian, is_measurement=True).compose(state)
expectation = AerPauliExpectation().convert(measurable_expression)
sampler = CircuitSampler(qi).convert(expectation)
value=sampler.eval()
print("Energy at x=3: %f"%(value+nuc_rep))
from qiskit.quantum_info import Statevector
out=Statevector(UCC_2_ansatz)
for k,elem in enumerate(out):
    if abs(elem)>1e-6:
        print(format(k,"b").zfill(num_spin_orbitals), " ", np.real(elem))
"""
###
for k,sample_x_val in enumerate(sample_x): #Get circuits
    print("Baemp")
    idx = (np.abs(xvals - sample_x_val)).argmin()
    x=xvals[idx]
    print(x)
    UCC_2_param=UCC_2_params[idx]
    UCC_2_ansatz_f,initial_point=UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter,reps=1,generalized=False)
    UCC_2_ansatz=UCC_2_ansatz_f.assign_parameters(UCC_2_param)
    #UCC_2_ansatz=HartreeFock(num_spin_orbitals=num_spin_orbitals,num_particles=num_particles,qubit_converter=qubit_converter)
    #Trasform to procrustes basis
    procrustes_state,R=localize_procrustes(mol,mo_coeff_min[idx],mf.mo_occ,ref_det,mix_states=True,return_R=True,active_orbitals=active_space,nelec=nelec)
    R=np.linalg.inv(mo_coeff_min[idx])@procrustes_state
    R=R[1:,1:]

    n_qubits=len(active_space)
    qc=QuantumCircuit(n_qubits*2)
    qc.append(UCC_2_ansatz,list(np.arange(n_qubits*2)))
    gate=get_transformation_circuit(R,active_space,nelec)
    qc.append(gate,list(np.arange(n_qubits)))
    qc.append(gate,list(np.arange(n_qubits,2*n_qubits)))

    UCC_2_circuits.append(qc)
print("Done getting ciruicts")
for i in range(len(UCC_2_circuits)):
    print("bæmp")
    newcirc=transpile(UCC_2_circuits[i],backend=backend,optimization_level=1)
    UCC_2_circuits[i]=newcirc
print("Done transforming circuits")
zero1states={}
pauli_string_exp_vals={}
overlaps={}
for i in range(len(sample_x)):
    for j in range(i,len(sample_x)):
        zero1states["%d%d"%(i,j)]=get_01_state(UCC_2_circuits[i],UCC_2_circuits[j],num_qubits,backend)
print("Done getting new states")
print(zero1states)
irreps=[]
mo_coeff_min=[]
energies=np.zeros((len(x_of_interest),3))
for k,x in enumerate(x_of_interest):
    print(x)
    atom=molecule(x)
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


E_EVC=np.zeros(len(x_of_interest))
E_exact=np.zeros(len(x_of_interest))
S=np.zeros((len(sample_x),len(sample_x)))
for i in range(len(sample_x)):
    for j in range(i,len(sample_x)):
        S[i,j]=S[j,i]=get_overlap_expectations(zero1states["%d%d"%(i,j)],num_qubits,qi)
for k,x in enumerate(x_of_interest):
    print(k)
    H=np.zeros((len(sample_x),len(sample_x)))

    procrustes_state,R=localize_procrustes(mol,mo_coeff_min[k],mf.mo_occ,ref_det,mix_states=True,return_R=True,active_orbitals=active_space,nelec=nelec)
    R=R[1:,1:]
    procrastes_hamiltonian, num_particles,num_spin_orbitals, nuc_rep,newGroup=get_basis_Hamiltonian(molecule,x,qi,procrustes_state,basis=basis,active_space=active_space,nelec=nelec,symmetry='C2v',irreps=irreps[k])
    qubit_hamiltonian=qubit_converter.convert(procrastes_hamiltonian).reduce()
    for i in range(len(sample_x)):
        for j in range(i,len(sample_x)):
            s=S[i,j]
            h=get_energy_expectations_01state(zero1states["%d%d"%(i,j)],num_qubits,qubit_hamiltonian,qi)
            H[i,j]=H[j,i]=np.real(h+s*nuc_rep)
            print(h,s)
    e,cl,c=eig(scipy.linalg.pinv(S,atol=1e-8)@H,left=True)
    idx = np.real(e).argsort()
    e = e[idx]
    c = c[:,idx]
    cl = cl[:,idx]
    E_EVC[k]=np.real(e[0])
    print(E_EVC[k],E_FCI[k])
plt.plot(x_of_interest,E_EVC,label="EVC (0,1,2,3,4)")
plt.plot(x_of_interest,E_FCI,label="exact")
plt.legend()
plt.ylabel("Energy (Hartree)")
plt.xlabel("x (Bohr)")
plt.tight_layout()
plt.savefig("BeH2_difficult.pdf")
plt.show()
