# Master thesis
## Simon Elias Schrader, UiO, 2021/2022

This is the GitHub page for the code used in the master thesis in Computational Science: Chemistry at UiO of Simon Elias Schrader, spring 2022.

The folder structure is as follows:
- The `tex` folder contains the data to create the .pdf of the thesis
- The `code` folder contains all code created by us.
  - Inside of the `code` folder, the folder `systems` contains the following folders:
    - `coupled_cluster` contains files to create the plots and tables in chapter 7, e.g. WF-CCEVC and AMP-CCEVC.
    - `HF_geom_groundstate`, `HF_geom_ExcitedStates`, `HF_geom_tweakRepulsion` contain files to create the plots and tables in chapter 6.
    - `quantum_computing` contains files to create the plots and tables in chapter 9.
    - `matrix_system` contains files to create the plots with the EVC example (fig 2.1).
    - `concepts` contains files to create the other plots in the theory and method chapters.
    - `libraries` contains the "back bone" of the thesis, containing many of the necessary algorithms to perform EVC. 

In order to run our code, `pyscf` 2.0.1 , `qiskit` 0.20.0, `qiskit_nature` 0.3.2,  `openFermion`1.4.0.dev0 and `opt_einsum`v3.3.0 need to be installed. In addition, [quantum systems](https://github.com/Schoyen/quantum-systems) by Ø. Schøyen is required, as well as
Ø. Schøyen's coupled cluster code, which is private. If Schøyen's CC code is available, the file `coupled_cluster/rccsd/rhs_t.py` needs to be replaced with `code/cc_replace/rhs_t.py` in this repo.
