import sys
sys.path.append("../libraries")
from quantum_library import *
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
x_of_interest=np.linspace(1.5,6.5,51)
E_HEA=np.zeros(len(x_of_interest))
backend= Aer.get_backend('statevector_simulator')
seed=np.random.randint(100000)
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

optimizer=SciPyOptimizer("BFGS")
numPyEigensolver=NumPyEigensolver()

qubit_converter_nosymmetry = QubitConverter(mapper=JordanWignerMapper(),two_qubit_reduction=False)
#tapering_value=[1,1,1]
#qubit_converter_symmetry=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True,z2symmetry_reduction=tapering_value)
qubit_converter_symmetry=qubit_converter_nosymmetry
param_list=[]
E_exact=np.zeros(len(x_of_interest))
E_HEA=np.zeros(len(x_of_interest))
optimal_params=None
for k,x in enumerate(x_of_interest):
    print(k)
    hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,x,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
    qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
    num_qubits=qubit_hamiltonian.num_qubits
    result = numPyEigensolver.compute_eigenvalues(qubit_hamiltonian)
    E_exact[k]=np.real(result.eigenvalues[0]+nuc_rep)
    ansatz,initial_point=UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=1,generalized=False)
    initial_point=np.zeros(len(initial_point))
    if optimal_params is None:
        optimal_params=initial_point
    unitary,energy,optimal_params=get_unitary(qubit_hamiltonian,ansatz,optimizer,qi,include_custom=True,initial_point=optimal_params,nuc_rep=nuc_rep)
    param_list.append(optimal_params)
    E_HEA[k]=energy
    print("Exact: %f, HEA: %f"%(E_exact[k],energy))
dicterino={}
dicterino["xvals"]=x_of_interest
dicterino["circuits"]=param_list
dicterino["UCCSD2"]=E_HEA
savemat("data/BeH2_Jordanwigner_UCCSD2.mat",dicterino)
