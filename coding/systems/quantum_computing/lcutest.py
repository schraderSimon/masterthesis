import matplotlib.pyplot as plt
from measure_overlap import *

qc=QuantumCircuit()
num_qubits=1
qr=QuantumRegister(num_qubits,"q")
qc.add_register(qr)
unitary1=circuit_to_gate(qc)
qc1=QuantumCircuit()
qr=QuantumRegister(num_qubits,"q")
qc1.add_register(qr)
qc1.x(0)
unitary2=circuit_to_gate(qc1)
qc2=QuantumCircuit()
num_qubits=1
qr=QuantumRegister(num_qubits,"q")
qc2.add_register(qr)
qc2.h(0)
unitary1=circuit_to_gate(qc2)

left_unitaries=[qc1,qc]
right_unitaries=[qc1,qc]
left_parameters=[0.7,0.9]
right_parameters=[1,0.3]
backend=Aer.get_backend("statevector_simulator")
#standard_LCU(left_unitaries,left_parameters,num_qubits,backend=backend)
LCU_2(left_unitaries,right_unitaries,left_parameters,right_parameters,num_qubits,backend=backend)
left_state=Statevector(left_unitaries[0])*left_parameters[0]
for i in range(1,len(left_unitaries)):
    left_state=left_state+Statevector(left_unitaries[i])*(left_parameters[i])
left_state=left_state/np.sqrt(left_state.inner(left_state))

right_state=Statevector(right_unitaries[0])*(right_parameters[0])
for i in range(1,len(right_unitaries)):
    right_state=right_state+Statevector(right_unitaries[i])*(right_parameters[i])
right_state=right_state/np.sqrt(right_state.inner(right_state))

print("Overlap:")
print(left_state.inner(right_state))
