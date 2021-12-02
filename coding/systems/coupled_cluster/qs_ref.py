import warnings
import scipy
import numpy as np
from quantum_systems import (
    BasisSet,
    SpatialOrbitalSystem,
    GeneralOrbitalSystem,
    QuantumSystem,
)
def orthogonal_procrustes(mo_new,reference_mo):
    A=reference_mo.T
    B=mo_new.T
    M=B@A.T
    U,s,Vt=scipy.linalg.svd(M)
    return U@Vt, 0
def localize_procrustes(mol,mo_coeff,mo_occ,ref_mo_coeff,mix_states=False):
    """Performs the orthgogonal procrustes on the occupied and the unoccupied molecular orbitals.
    ref_mo_coeff is the mo_coefs of the reference state.
    If "mix_states" is True, then mixing of occupied and unoccupied MO's is allowed.
    """
    if mix_states==False:
        mo=mo_coeff[:,mo_occ>0]
        premo=ref_mo_coeff[:,mo_occ>0]
        R,scale=orthogonal_procrustes(mo,premo)
        mo=mo@R

        mo_coeff[:,mo_occ>0]=np.array(mo)
        mo_unocc=mo_coeff[:,mo_occ<=0]
        premo=ref_mo_coeff[:,mo_occ<=0]
        R,scale=orthogonal_procrustes(mo_unocc,premo)
        mo_unocc=mo_unocc@R

        mo_coeff[:,mo_occ<=0]=np.array(mo_unocc)


    elif mix_states==True:
        mo=mo_coeff[:,:]
        premo=ref_mo_coeff[:,:]
        R,scale=orthogonal_procrustes(mo,premo)
        mo=mo@R

        mo_coeff[:,:]=np.array(mo)
    return mo_coeff
def construct_pyscf_system_rhf_ref(
    molecule,
    basis="cc-pvdz",
    add_spin=True,
    anti_symmetrize=True,
    np=None,
    verbose=False,
    charge=0,
    cart=False,
    reference_state=None,
    **kwargs,
):
    """Convenience function setting up a closed-shell atom or a molecule from
    PySCF as a ``QuantumSystem`` in RHF-basis using PySCF's RHF-solver.
    Parameters
    ----------
    molecule : str
        String describing the atom or molecule. This gets passed to PySCF which
        means that we support all the same string options as PySCF.
    basis : str
        String describing the basis set. PySCF determines which options are
        available.
    add_spin : bool
        Whether or not to return a ``SpatialOrbitalSystem`` (``False``) or a
        ``GeneralOrbitalSystem`` (``True``). Default is ``True``.
    anti_symmetrize : bool
        Whether or not to anti-symmetrize the two-body elements in a
        ``GeneralOrbitalSystem``. This only applies if ``add_spin = True``.
        Default is ``True``.
    np : module
        Array- and linear algebra module.
    Returns
    -------
    SpatialOrbitalSystem, GeneralOrbitalSystem
        Depending on the choice of ``add_spin`` we return a
        ``SpatialOrbitalSystem`` (``add_spin = False``), or a
        ``GeneralOrbitalSystem`` (``add_spin = True``).
    See Also
    -------
    PySCF
    Example
    -------
    >>> # Set up the Beryllium atom centered at (0, 0, 0)
    >>> system = construct_pyscf_system_rhf(
    ...     "be 0 0 0", basis="cc-pVDZ", add_spin=False
    ... ) # doctest.ELLIPSIS
    converged SCF energy = -14.5723...
    >>> # Compare the number of occupied basis functions
    >>> system.n == 4 // 2
    True
    >>> gos = system.construct_general_orbital_system()
    >>> gos.n == 4
    True
    >>> system = construct_pyscf_system_rhf(
    ...     "be 0 0 0", basis="cc-pVDZ"
    ... ) # doctest.ELLIPSIS
    converged SCF energy = -14.5723...
    >>> system.n == gos.n
    True
    """

    import pyscf

    if np is None:
        import numpy as np

    # Build molecule in AO-basis
    mol = pyscf.gto.Mole()
    mol.unit = "bohr"
    mol.charge = charge
    mol.cart = cart
    mol.build(atom=molecule, basis=basis, **kwargs)
    nuclear_repulsion_energy = mol.energy_nuc()

    n = mol.nelectron
    assert (
        n % 2 == 0
    ), "We require closed shell, with an even number of particles"

    l = mol.nao

    hf = pyscf.scf.RHF(mol)
    hf_energy = hf.kernel()

    if not hf.converged:
        warnings.warn("RHF calculation did not converge")

    if verbose:
        print(f"RHF energy: {hf.e_tot}")

    charges = mol.atom_charges()
    coords = mol.atom_coords()
    nuc_charge_center = np.einsum("z,zx->x", charges, coords) / charges.sum()
    mol.set_common_orig_(nuc_charge_center)
    if reference_state is None:
        C = np.asarray(hf.mo_coeff)
    else:
        C=localize_procrustes(mol,hf.mo_coeff,hf.mo_occ,ref_mo_coeff=reference_state,mix_states=False)
    h = pyscf.scf.hf.get_hcore(mol)
    s = mol.intor_symmetric("int1e_ovlp")
    u = mol.intor("int2e").reshape(l, l, l, l).transpose(0, 2, 1, 3)
    position = mol.intor("int1e_r").reshape(3, l, l)

    bs = BasisSet(l, dim=3, np=np)
    bs.h = h
    bs.s = s
    bs.u = u
    bs.nuclear_repulsion_energy = nuclear_repulsion_energy
    bs.particle_charge = -1
    bs.position = position
    bs.change_module(np=np)

    system = SpatialOrbitalSystem(n, bs)
    system.change_basis(C)

    return (
        system.construct_general_orbital_system(anti_symmetrize=anti_symmetrize)
        if add_spin
        else system
    )
