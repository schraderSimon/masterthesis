from measure_overlap import *

def molecule(x):
    return "Be 0 0 0; H 0 0 %f; H 0 0 -%f"%(x,x)
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from optimparallel import minimize_parallel
"""
def multi_run_wrapper(args):
   return cost_function(*args)

def cost_function(x,previous_states,ansatz,vqe,qubit_hamiltonian):
    energy=vqe_cost=vqe.get_energy_evaluation(qubit_hamiltonian)(x)
    print("Energy: %f"%energy)
    overlapSquared=0
    price=0
    stateVec=Statevector(ansatz.assign_parameters(x))
    for i in range(len(previous_states)):
        ovlp=stateVec.inner(previous_states[i])
        price+=abs(ovlp)**2*overlap_params[i]
    energy+=price
    print("Price: %f"%price)
    return energy
def derivative_function(x,previous_states,ansatz,vqe,qubit_hamiltonian):
    zero_pred=cost_function(x,previous_states,ansatz,vqe,qubit_hamiltonian)
    derivparams = []
    delta=1e-8
    for i in range(len(x)):
        copy = np.array(x)
        copy[i] += delta
        derivparams.append(copy)
    data=[[derivparams[i],previous_states,ansatz,vqe,qubit_hamiltonian] for i in range(len(x))]
    results=pool.map(multi_run_wrapper, data)
    derivs = np.array([ (r - zero_pred)/delta for r in results ])
    return derivs

def VQCES_algorithm_parallel(qubit_hamiltonian,prev_params,previous_unitaries,overlap_params,ansatz,initial_point,nuc_rep,num_qubits,qi,method="BFGS"):
    previous_states=[]
    vqe = VQE(ansatz=ansatz,include_custom=True,quantum_instance=qi)
    for unitary in previous_unitaries:
        previous_states.append(Statevector(unitary))
    print("Running VQCES_algorithm")
    res=minimize(cost_function, initial_point,args=(previous_states,ansatz,vqe,qubit_hamiltonian),jac=derivative_function,method="BFGS")
    print("Done VQCES_algorithm")
    best_unitary=ansatz.assign_parameters(res.x)
    state=CircuitStateFn(best_unitary)
    energy_measurement=qubit_hamiltonian
    measurable_expression = StateFn(energy_measurement, is_measurement=True).compose(state)
    expectation = AerPauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(qi).convert(expectation)
    energy=sampler.eval()
    return best_unitary, energy, res.x
"""
def VQCES_algorithm(qubit_hamiltonian,prev_params,previous_unitaries,overlap_params,ansatz,initial_point,nuc_rep,num_qubits,qi,method="BFGS"):
    previous_states=[]
    vqe = VQE(ansatz=ansatz,include_custom=True,quantum_instance=qi)
    vqe_cost=vqe.get_energy_evaluation(qubit_hamiltonian)
    for unitary in previous_unitaries:
        previous_states.append(Statevector(unitary))
    def cost_function(x):
        energy=vqe_cost(x)
        overlapSquared=0
        price=0
        stateVec=Statevector(ansatz.assign_parameters(x))
        for i in range(len(prev_params)):
            ovlp=stateVec.inner(previous_states[i])
            price+=abs(ovlp)**2*overlap_params[i]
        energy+=price
        return energy
    print("Running VQCES_algorithm")
    res=minimize(cost_function, initial_point,method="BFGS")
    print("Done VQCES_algorithm")
    best_unitary=ansatz.assign_parameters(res.x)
    state=CircuitStateFn(best_unitary)
    energy_measurement=qubit_hamiltonian
    measurable_expression = StateFn(energy_measurement, is_measurement=True).compose(state)
    expectation = AerPauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(qi).convert(expectation)
    energy=sampler.eval()
    return best_unitary, energy, res.x
import multiprocessing
from itertools import repeat

from dask.distributed import Client
if __name__=="__main__":
    basis="STO-6G"
    exc = Client()
    ref_x=2
    mol = mol = gto.M(atom=molecule(ref_x), basis=basis,unit="Bohr")
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    hcore_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    nuc_rep=mf.energy_nuc()
    ref_det=mf.mo_coeff
    sample_x=[1.5,3.5,4.5,5.5,6.5]
    E_EVC=np.zeros(len(sample_x))
    E_exact=np.zeros(len(sample_x))
    E_UCC=np.zeros(len(sample_x))
    E_k2=np.zeros(len(sample_x))
    backend = QasmSimulator(method='statevector')

    backend.set_options(executor=exc, max_job_size=1,statevector_parallel_threshold=7,statevector_sample_measure_opt=7)
    seed=np.random.randint(100000)
    qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

    numPyEigensolver=NumPyEigensolver(k=3)

    qubit_converter_nosymmetry = QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True)
    tapering_value=[1,1,1]
    qubit_converter_symmetry=QubitConverter(mapper=ParityMapper(),two_qubit_reduction=True,z2symmetry_reduction=tapering_value)
    optimal_params=None
    #pool=multiprocessing.Pool(7)
    energies=np.zeros(len(sample_x))
    E_exact=np.zeros(len(sample_x))
    active_space=[1,2,3,4,5,6]
    nelec=4
    previous_unitaries=[]
    overlap_params=[3]*(len(sample_x)-1)
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
        ansatz,initial_point=UCC_ansatz(num_particles,num_spin_orbitals,num_qubits,qubit_converter=qubit_converter_symmetry,reps=2,generalized=False)
        unitary,energy,optimal_params=VQCES_algorithm(qubit_hamiltonian,params,previous_unitaries,overlap_params,ansatz,initial_point,nuc_rep,num_qubits,qi,method="BFGS")
        energy+=nuc_rep
        previous_unitaries.append(unitary)
        params.append(optimal_params)
        print("%f: %.4f, %.4f"%(x,energy,e_exact))
        print(params[-1])
    dicterino={}
    dicterino["xvals"]=sample_x
    dicterino["UCC_2"]=params
    savemat("energy_data/BeH2_params_EXCITED_UCC2_new.mat",dicterino)
