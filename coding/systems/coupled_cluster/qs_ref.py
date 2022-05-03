import sys
sys.path.append("/home/simon/Documents/University/masteroppgave/coding/systems/libraries")
from func_lib import *
from quantum_systems import (
    BasisSet,
    SpatialOrbitalSystem,
    GeneralOrbitalSystem,
    QuantumSystem,
)
import pyscf
def construct_pyscf_system_rhf_ref(
    molecule,
    basis="cc-pvdz",
    add_spin=True,
    anti_symmetrize=False,
    np=None,
    verbose=False,
    charge=0,
    cart=False,
    reference_state=None,
    mix_states=False,
    return_C=False,
    weights=None,
    givenC=None,
    truncation=1000000,
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
    print(molecule)
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
    if reference_state is None and givenC is None:
        C = np.asarray(hf.mo_coeff)
    elif givenC is None:
        C=localize_procrustes(mol,hf.mo_coeff,hf.mo_occ,ref_mo_coeff=reference_state,mix_states=mix_states,weights=weights)
    elif reference_state is None:
        C=givenC
        print("Use given C")
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
    #system.change_basis(C)
    system.change_basis(C[:,:truncation])
    if return_C:
        return (
            system.construct_general_orbital_system(anti_symmetrize=anti_symmetrize)
            if add_spin
            else system
        ), C
    return (
        system.construct_general_orbital_system(anti_symmetrize=anti_symmetrize)
        if add_spin
        else system
    )

def construct_pyscf_system_rhf_natorb(
    molecule,
    basis="cc-pvdz",
    add_spin=True,
    anti_symmetrize=False,
    np=None,
    verbose=False,
    charge=0,
    cart=False,
    reference_natorbs=None,
    reference_noons=None,
    reference_overlap=None,
    return_natorbs=True,
    weights=None,
    truncation=1000000,
    **kwargs,
):
    if np is None:
        import numpy as np
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
    if reference_noons is None:
        mymp=pyscf.mp.RMP2(hf).run()
        new_noons,C=pyscf.mcscf.addons.make_natural_orbitals(mymp)
    else:
        mymp=pyscf.mp.RMP2(hf).run()
        noons,natorbs=pyscf.mcscf.addons.make_natural_orbitals(mymp)
        new_noons,C=similiarize_natural_orbitals(reference_noons,reference_natorbs,noons,natorbs,mol.nelec,mol.intor("int1e_ovlp"),reference_overlap)
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
    system.change_basis(C[:,:truncation])
    if return_natorbs:
        return (
            system.construct_general_orbital_system(anti_symmetrize=anti_symmetrize)
            if add_spin
            else system
        ), C, new_noons, mol.intor("int1e_ovlp")
    return (
        system.construct_general_orbital_system(anti_symmetrize=anti_symmetrize)
        if add_spin
        else system
    )
def construct_pyscf_system_rhf_canonicalorb(
    molecule,
    basis="cc-pvdz",
    add_spin=True,
    anti_symmetrize=False,
    np=None,
    verbose=False,
    charge=0,
    cart=False,
    reference_natorbs=None,
    reference_noons=None,
    reference_overlap=None,
    return_natorbs=True,
    weights=None,
    truncation=100000,
    **kwargs,
):
    if np is None:
        import numpy as np
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
    if reference_noons is None:
        mo_coeff=hf.mo_coeff
        mo_energy=hf.mo_energy
        new_noons=mo_energy
        C=mo_coeff
    else:
        natorbs=mo_coeff=hf.mo_coeff
        noons=mo_energy=hf.mo_energy
        new_noons,C=similiarize_canonical_orbitals(reference_noons,reference_natorbs,noons,natorbs,mol.nelec,mol.intor("int1e_ovlp"),reference_overlap)
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
    system.change_basis(C[:,:truncation])
    if return_natorbs:
        return (
            system.construct_general_orbital_system(anti_symmetrize=anti_symmetrize)
            if add_spin
            else system
        ), C, new_noons, mol.intor("int1e_ovlp")
    return (
        system.construct_general_orbital_system(anti_symmetrize=anti_symmetrize)
        if add_spin
        else system
    )




def construct_pyscf_system_ghf_ref(
    molecule,
    basis="cc-pvdz",
    add_spin=True,
    anti_symmetrize=False,
    np=None,
    verbose=False,
    charge=0,
    cart=False,
    reference_state=None,
    mix_states=False,
    return_C=False,
    weights=None,
    givenC=None,
    truncation=1000000,
    **kwargs,
):
    import pyscf

    if np is None:
        import numpy as np

    # Build molecule in AO-basis
    mol = pyscf.gto.Mole()
    mol.unit = "bohr"
    mol.charge = charge
    mol.cart = cart
    print(molecule)
    mol.build(atom=molecule, basis=basis, **kwargs)
    nuclear_repulsion_energy = mol.energy_nuc()
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
    if reference_state is None and givenC is None:
        C = np.asarray(hf.mo_coeff)
    elif givenC is None:
        C=localize_procrustes(mol,hf.mo_coeff,hf.mo_occ,ref_mo_coeff=reference_state,mix_states=mix_states,weights=weights)
    elif reference_state is None:
        C=givenC
        print("Use given C")
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
    #system.change_basis(C)
    system.change_basis(C[:,:truncation])
    if return_C:
        return (
            system.construct_general_orbital_system(anti_symmetrize=anti_symmetrize)
            if add_spin
            else system
        ), C
    return (
        system.construct_general_orbital_system(anti_symmetrize=anti_symmetrize)
        if add_spin
        else system
    )
