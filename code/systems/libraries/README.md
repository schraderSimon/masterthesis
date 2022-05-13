This folder contains the "backbone" of the thesis, e.g. the methods and classes to implement eigenvector continuation.

- `func_lib.py`, `matrix_operations.py` and `helper_functions.py` contain much-used convenience functions, such as the Procrustes algorithm, canonical orthonormalization, algorithms to ascertain the analyticity of NOs, functions to obtain FCI/CCSD/RHF energies for a list of geometries etc.
- `rccsd_gs.py` contains the necessary algorithms & data to implement AMP-CCEVC (and the parameter-reduced form) and WF-CCEVC using RHF determinants and spin-adapted CCSD
- `general_CCSD_GS.py` contains the necessary algorithms & data to implement WF-CCEVC using RHF determinants and general CCSD
- `quantum_library.py` contains algorithms for EVC on a quantum computer.
- `qs_ref.py` contains methods to generate orbitalSystems in Schøyens code, adapted in such a way that Procrustes orbitals (or natural orbitals)
- `REC.py` contains methods and classes to perform Multi-reference EVC with Slater determinants
- `rhs_t.py` contains the code for restricted CCSD amplitude calculations. It also contains code for parameter-reduced AMP-CCEVC, where not all parameters are included. 
