from measure_overlap import *

np.set_printoptions(linewidth=300,precision=10,suppress=True)
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
x_of_interest=np.linspace(1.5,6.5,11)
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
backend= QasmSimulator(method='statevector')
seed=np.random.randint(100000)
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

optimizer=SciPyOptimizer("BFGS")#L_BFGS_B()
numPyEigensolver=NumPyEigensolver()

tapering_value=[1,1,1]
qubit_converter=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True,z2symmetry_reduction=tapering_value)
UCC_params_l=loadmat("BeH2_params_EXCITED_UCC2_power8_overlapParam3.mat")
UCC_2_params=UCC_params_l["UCC_2"]
xvals=UCC_params_l["xvals"][0][:]
#sample_x=[0,1,2,3,4]
sample_x=xvals
UCC_circuits=[]
hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,2,qi,active_space=active_space,nelec=nelec,basis=basis)
qubit_hamiltonian=qubit_converter.convert(hamiltonian,num_particles=num_particles)
#qubit_hamiltonian=qubit_hamiltonian.reduce()
num_qubits=qubit_hamiltonian.num_qubits


UCC_UCC2_params=loadmat("BeH2_UCC_vals.mat")
UCC_2_params_full=UCC_UCC2_params["UCC_2"]
xvals_UCC1=UCC_UCC2_params["xvals"][0]
full_UCC_2_circuits=[]
for sample_x_val in xvals: #Get circuits
    print("Baemp")
    idx = (np.abs(xvals - sample_x_val)).argmin()
    x=xvals[idx]
    UCC_param=UCC_2_params[idx]
    UCC_ansatz_f,initial_point=UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter,reps=2,generalized=False)
    UCC_circuits.append(UCC_ansatz_f.assign_parameters(UCC_param))
    print(UCC_param)
print("Done getting ciruicts")
for i in range(len(UCC_circuits)):
    print("bæmp")
    newcirc=transpile(UCC_circuits[i],backend=backend,optimization_level=1)
    UCC_circuits[i]=newcirc
print("Done transforming circuits")
zero1states={}
pauli_string_exp_vals={}
overlaps={}
for i in range(len(sample_x)):
    for j in range(i,len(sample_x)):
        zero1states["%d%d"%(i,j)]=get_01_state(UCC_circuits[i],UCC_circuits[j],num_qubits,backend)
print("Done getting new states")
energies=np.zeros((len(x_of_interest),3))
E_EVC=np.zeros(len(x_of_interest))
E_exact=np.zeros(len(x_of_interest))
S=np.zeros((len(sample_x),len(sample_x)))
for i in range(len(sample_x)):
    for j in range(i,len(sample_x)):
        S[i,j]=S[j,i]=get_overlap_expectations(zero1states["%d%d"%(i,j)],num_qubits,qi)
print(S)
print(np.linalg.inv(S))
print(np.linalg.eig(S)[0])
for k,x in enumerate(x_of_interest):
    print(k)
    H=np.zeros((len(sample_x),len(sample_x)))
    hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,x,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
    qubit_hamiltonian=qubit_converter.convert(hamiltonian,num_particles=num_particles)
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
    energy_exact=np.real(numPyEigensolver.compute_eigenvalues(qubit_hamiltonian).eigenvalues[0])
    E_FCI[k]=energy_exact+nuc_rep
    print(H[0,0],energy_exact)
    print(E_EVC[k],E_FCI[k])
"""
for sample_x_val in xvals_UCC1: #Get circuits
    print("Baemp")
    idx = (np.abs(xvals_UCC1 - sample_x_val)).argmin()
    UCC_param=UCC_2_params_full[idx]
    UCC_ansatz_f,initial_point=UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter,reps=2,generalized=False)
    full_UCC_2_circuits.append(UCC_ansatz_f.assign_parameters(UCC_param))
E_UCC=np.zeros(len(xvals_UCC1))
for k,x in enumerate(xvals_UCC1):
    hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,x,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
    qubit_hamiltonian=qubit_converter.convert(hamiltonian,num_particles=num_particles)
    state=CircuitStateFn(full_UCC_2_circuits[k])
    energy_measurement=qubit_hamiltonian
    measurable_expression = StateFn(energy_measurement, is_measurement=True).compose(state)
    expectation = AerPauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(qi).convert(expectation)
    E_UCC[k]=sampler.eval()+nuc_rep
    print(E_UCC[k],numPyEigensolver.compute_eigenvalues(qubit_hamiltonian).eigenvalues[0]+nuc_rep)
"""
plt.plot(x_of_interest,E_EVC,label="noS-EVC")
plt.plot(x_of_interest,E_FCI,label="exact")
plt.legend()
plt.ylabel("Energy (Hartree)")
plt.xlabel("x (Bohr)")
plt.tight_layout()
plt.savefig("BeH2_difficult_VQE_UCC2.pdf")
plt.show()
