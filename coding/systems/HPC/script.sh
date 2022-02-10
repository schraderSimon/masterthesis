# Set up job environment
set -o errexit # exit on any error
set -o nounset # treat unset variables as error

# Load modules
module load Python/3.8.6-GCCcore-10.2.0

# Set the ${PS1} (needed in the source of the virtual environment for some Python versions)
export PS1=\$

# activate the virtual environment
source qiskit_pyscf_env/bin/activate

# execute example script
python VQCES_BeH2.py
