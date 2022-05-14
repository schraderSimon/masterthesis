# Master thesis
## Simon Elias Schrader, UiO, 2021/2022
https://github.com/schraderSimon/masterthesis

This is the GitHub page for the code used in Simon Elias Schrader's master thesis in Computational Science: Chemistry at the University of Oslo, spring 2022.

The folder structure is as follows:
- The `code` folder contains all code created by us.
  - Inside of the `code` folder, the folder `systems` contains the following folders:
    - `coupled_cluster` contains files to create the plots and tables in chapter 7, e.g. WF-CCEVC and AMP-CCEVC.
    - `HF_geom_groundstate`, `HF_geom_ExcitedStates`, `HF_geom_tweakRepulsion` contain files to create the plots and tables in chapter 6.
    - `quantum_computing` contains files to create the plots and tables in chapter 8.
    - `matrix_system` contains files to create the plots with the EVC example (fig 2.1).
    - `concepts` contains files to create the other plots in the theory and method chapters. It has it's own readme.
    - `libraries` contains the "back bone" of the thesis, containing many of the necessary algorithms to perform EVC. It has it's own Readme.
    - `test` contains a single test file, which runs some basic tests to test that our algorithms work as they should (EVC methods being approximately exact or better at sample geometries; the correct ordering of natural orbitals, the continuity of Procrustes orbitals).


In order to run our code, `pyscf` 2.0.1 , `qiskit` 0.20.0, `qiskit_nature` 0.3.2,  `openFermion`1.4.0.dev0 and `opt_einsum`v3.3.0 need to be installed. In addition, [quantum systems](https://github.com/Schoyen/quantum-systems) by Ø. Schøyen is required, as well as
Ø. Schøyen's coupled cluster code, which is private.
In general, we think that the file names are quite self explanatory. However, for the `concepts` and `libraries` folders, we added readme files that describe (briefly) what is contained in the individual files.
