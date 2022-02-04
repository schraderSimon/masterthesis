from measure_overlap import *
from pyscf import gto, scf, ao2mo, fci, mcscf
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, cc,scf, ao2mo,fci
import openfermion, cirq
def molecule(x):
    return "Be 0 0 0; H 0 0 %f; H 0 0 -%f"%(x,x)
def molecule(x):
    return "H 0 0 0; H 0 0 %f; H 0 0 %f; H 0 0 %f"%(x,2*x,3*x)
basis="STO-6G"
ref_x=2
mol = mol = gto.M(atom=molecule(ref_x), basis=basis,unit="Bohr")
mol.build()
mf = scf.RHF(mol)
mf.kernel()
hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
nuc_rep=mf.energy_nuc()
ref_det=mf.mo_coeff
sample_x=[1,2,3,4,5,6]
E_EVC=np.zeros(len(sample_x))
E_exact=np.zeros(len(sample_x))
E_UCC=np.zeros(len(sample_x))
E_k2=np.zeros(len(sample_x))
backend= Aer.get_backend('statevector_simulator')
seed=np.random.randint(100000)
qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

numPyEigensolver=NumPyEigensolver(k=3)

qubit_converter_nosymmetry = QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True)
tapering_value=[1,1,1]
qubit_converter_symmetry=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True,z2symmetry_reduction=tapering_value)
optimal_params=None
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
def VQCES_algorithm(qubit_hamiltonian,prev_params,overlap_params,ansatz,initial_point,nuc_rep,num_qubits,qi,method="BFGS"):
    vqe = VQE(ansatz=ansatz,include_custom=True,quantum_instance=qi)
    inverse=ansatz.copy().inverse()
    new=ParameterVector("n",length=len(inverse.parameters))
    inverse=inverse.assign_parameters(new)
    qc=QuantumCircuit()
    qr=QuantumRegister(num_qubits,"q")
    qc.add_register( qr )
    qc.append(inverse.to_gate(),qr)
    qc.append(ansatz.to_gate(),qr)
    overlap_vqe=VQE(ansatz=qc,include_custom=True,quantum_instance=qi)
    vqe_cost=vqe.get_energy_evaluation(qubit_hamiltonian)
    overlap_measurement=I
    for i in range(num_qubits-1):
        overlap_measurement=overlap_measurement^I
    overlap_func=overlap_vqe.get_energy_evaluation(overlap_measurement)
    def cost_function(x):
        energy=vqe_cost(x)
        print("Energy: %f"%energy)
        overlapSquared=0
        price=0
        for i in range(len(prev_params)):
            ovlp=Statevector(qc.assign_parameters(np.concatenate((x,prev_params[i]))))[0]
            price+=np.abs(ovlp)**2*overlap_params[i]
        energy+=price
        print("Price: %f"%price)
        return energy
    res = minimize(cost_function, initial_point, method=method)
    best_unitary=ansatz.assign_parameters(res.x)
    state=CircuitStateFn(best_unitary)
    energy_measurement=qubit_hamiltonian
    measurable_expression = StateFn(energy_measurement, is_measurement=True).compose(state)
    expectation = AerPauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(qi).convert(expectation)
    energy=sampler.eval()
    return best_unitary, energy, res.x
states=["gs","es","sEs"]
energies=np.zeros(len(sample_x))
E_exact=np.zeros(len(sample_x))
active_space=[1,2,3,4,5,6]
nelec=4
previous_unitaries=[]
overlap_params=[1]*(len(sample_x)-1)
params=[]
for k,x in enumerate(sample_x):
    print(overlap_params)
    hamiltonian, num_particles,num_spin_orbitals,nuc_rep=get_hamiltonian(molecule,x,qi,active_space=active_space,nelec=nelec,basis=basis,ref_det=ref_det)
    qubit_hamiltonian=qubit_converter_symmetry.convert(hamiltonian,num_particles=num_particles)
    num_qubits=qubit_hamiltonian.num_qubits
    print(qubit_hamiltonian.num_qubits)
    result = numPyEigensolver.compute_eigenvalues(qubit_hamiltonian)
    e_exact=np.real(result.eigenvalues[0]+nuc_rep)
    E_exact[k]=e_exact
    ansatz,initial_point=SUCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=1,include_singles=True)
    unitary,energy,optimal_params=VQCES_algorithm(qubit_hamiltonian,params,overlap_params,ansatz,initial_point,nuc_rep,num_qubits,qi,method="BFGS")
    energy+=nuc_rep
    print(optimal_params)
    previous_unitaries.append(unitary)
    params.append(optimal_params)
    print("%f: %.4f, %.4f"%(x,energy,e_exact))
dicterino={}
dicterino["xvals"]=sample_x
dicterino["UCC_1"]=params
savemat("LiH_params.mat",dicterino)
