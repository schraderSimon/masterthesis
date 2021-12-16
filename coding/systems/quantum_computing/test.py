import numpy as np
import matplotlib.pyplot as plt
np.random.seed(999)
target_distr = np.random.rand(2)
# We now convert the random vector into a valid probability vector
target_distr /= sum(target_distr)
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
def get_var_form(params):
    qc = QuantumCircuit(1, 1) #Create a circuit with 1 qbit and 1 measuring  bit
    qc.u(params[0], params[1], params[2],0) #Perform "u"
    #qc.ry(params[0],0)
    qc.draw(output="mpl")
    qc.measure(0,0) #measure "u" into 0
    return qc
from qiskit import Aer, transpile, assemble
backend = Aer.get_backend("qasm_simulator")
NUM_SHOTS = 1000

def get_probability_distribution(counts):
    if len(counts) == 1:
        try:
            counts["0"]
            counts["1"]=0
        except:
            counts["0"]=0
    output_distr = [counts["0"]/NUM_SHOTS, counts["1"]/NUM_SHOTS]
    print(output_distr)
    return output_distr

def objective_function(params):
    # Obtain a quantum circuit instance from the paramters
    qc = get_var_form(params) #Create the quantun algorithm
    # Execute the quantum circuit to obtain the probability distribution associated with the current parameters
    t_qc = transpile(qc, backend) #Create the simulated quantum circuit
    qobj = assemble(t_qc, shots=NUM_SHOTS) #assemble it with num_shots repetitions
    result = backend.run(qobj).result() #run the experiment
    # Obtain the counts for each measured state, and convert those counts into a probability vector
    print(params[0],result.get_counts(qc))
    output_distr = get_probability_distribution(result.get_counts(qc))

    # Calculate the cost as the distance between the output distribution and the target distribution
    cost = sum([np.abs(output_distr[i] - target_distr[i]) for i in range(2)])
    return cost
from qiskit.algorithms.optimizers import COBYLA, POWELL, SPSA, CG, ADAM, TNC, ESCH
# Initialize   optimizer
optimizer = COBYLA(maxiter=300)
# Create the initial parameters (noting that our single qubit variational form has 3 parameters)
params = np.random.rand(3)
ret = optimizer.optimize(num_vars=3, objective_function=objective_function, initial_point=params)

# Obtain the output distribution using the final parameters
qc = get_var_form(ret[0])
t_qc = transpile(qc, backend)
qobj = assemble(t_qc, shots=NUM_SHOTS*10)
counts = backend.run(qobj).result().get_counts(qc)
output_distr = get_probability_distribution(counts)

print("Target Distribution:", target_distr)
print("Obtained Distribution:", output_distr)
print("Output Error (Manhattan Distance):", ret[1])
print("Parameters Found:", ret[0])
