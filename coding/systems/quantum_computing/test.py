from qiskit import Aer
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import ElectronicStructureDriverType, ElectronicStructureMoleculeDriver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
import numpy as np
qubit_converter = QubitConverter(JordanWignerMapper())
from qiskit.algorithms import NumPyMinimumEigensolver

numpy_solver = NumPyMinimumEigensolver()
from qiskit_nature.algorithms import GroundStateEigensolver

calc = GroundStateEigensolver(qubit_converter, numpy_solver)
xs=np.linspace(2,3,11)

def molecule(x):
    y = lambda x: 2.54 - 0.46*x
    geometry=[['H', [-y(x)*0.529177249, 0., x*0.529177249]], ['H', [y(x)*0.529177249,0,0.529177249*x]],['Be', [0., 0., 0.]]]
    return Molecule(geometry=geometry,
                     charge=0)


for x in xs:
    print(x)
    driver = ElectronicStructureMoleculeDriver(molecule(x), basis='sto-3g', driver_type=ElectronicStructureDriverType.PSI4)
    es_problem = ElectronicStructureProblem(driver)
    res = calc.solve(es_problem)
    print(res)
