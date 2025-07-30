from tblite.interface import Calculator
import numpy as np
import numpy.linalg as LA

elements_atomnumber_dict = dict(
    zip(np.loadtxt(f"/home/nikolai/OrbitAll/examples/01_create_orbital_features/elements.txt", dtype=str), np.arange(118) + 1)
)

def _get_perm_map(element_numbers):
    """
    Permutation map for converting the matrices obtained from `tblite`
    GFN-xTB (closed shell) to the convention for qcore.

    Input (tblite) convention:
     - H: [1s, 2s]
     - C,N,O,F: [2s, 2py, 2pz, 2px]
    Output (qcore) convention:
     - H: [1s, 2s] (no change)
     - C,N,O,F: [2s, 2px, 2pz, 2py]
    """
    n_so = 0
    for el in element_numbers:
        if el == 1:
            n_so += 2
        else:
            n_so += 4

    perm_map = np.zeros(n_so, dtype=int)
    idx = 0
    for el in element_numbers:
        if el == 1:
            pmap = [idx, idx + 1]
            perm_map[idx : idx + 2] = pmap
            idx += 2
        else:
            # [s, py, pz, px] -> [s, px, pz, py]
            pmap = [idx, idx + 3, idx + 2, idx + 1]
            perm_map[idx : idx + 4] = pmap
            idx += 4
    return perm_map


def apply_perm_map(mat, perm_map):
    ndim = mat.ndim
    if ndim == 1:
        matp = mat[perm_map]
    elif ndim == 2:
        matp = mat[perm_map, :]
        matp = matp[:, perm_map]
    else:
        raise ValueError("Only 1D and 2D arrays are supported.")
    return matp


def read_xyz(xyzfile_path):
    n_atoms = np.loadtxt(xyzfile_path, max_rows=1, dtype=int)[()]
    atom_symbols = np.loadtxt(
        xyzfile_path, skiprows=2, usecols=[0], max_rows=n_atoms, dtype=str
    ).reshape((n_atoms,))
    element_numbers = np.array(
        [elements_atomnumber_dict[symbol] for symbol in atom_symbols]
    )

    coordinates = (
        np.loadtxt(
            xyzfile_path, skiprows=2, usecols=[1, 2, 3], max_rows=n_atoms
        ).reshape((n_atoms, 3))
        * 1.8897259886 # angstrom to bohr
    )

    return element_numbers, coordinates

def generate_xtb_features(
    element_numbers,
    coordinates,
    option="GFN1-xTB",
    charge=None,
    uhf=None,
    cpcm_epsilon=None, # for CPCM
    electronic_temperature=None,
    cutoff=None,
    spin_pol=False,
    get_energy=False,
    get_forces=False,
    get_dipole=False,
    get_partial_charges=False,
    # density_kernel=False,
    verbosity=False,
    **kwargs
):
    """
    Generates Fock matrix (F), density matrix (P), overlap matrix (S),
    and core Hamiltonian matrix (H) using the `tblite` library.

    Input:
    - element_numbers: List of atomic numbers of the atoms in the molecule.
    - coordinates: List of atomic coordinates in bohr units.
    - option (optional): xTB method to use. Default is "GFN1-xTB".
    - charge (optional): Charge of the molecule.
    - uhf (optional): Total electronic spin (NOT MULTIPLICITY!) of the molecule.
    - cutoff (optional): Cutoff value for the matrices. Values below this
        cutoff will be set to zero.
    - spin_pol (optional): If True, returns matrices for spin-polarized
        calculations. Default is False.
    - get_energy (optional): If True, returns the total energy of the molecule.
    - get_forces (optional): If True, returns the gradient of the energy

    Output:
    - If `spin_pol = True`, returns {F_a, F_b, P_a, P_b, S, H}
    - If `spin_pol = False`, returns {F, P, S, H}
    - If `get_energy = True`, adds `energy` field to the output.s
    - If `get_gradient = True`, adds `gradient` field to the output.
    - If `get_dipole = True`, adds `dipole` field to the output.
    - If `get_partial_charges = True`, adds `partial_charges` field to the output.
    """

    # Make sure to input coordinates with bohr units.
    calc = Calculator(
        method=option,
        numbers=element_numbers,
        positions=coordinates,
        charge=charge,
        uhf=uhf,
        **kwargs
    )
    if spin_pol:
        calc.add("spin-polarization", 1.0)
    if cpcm_epsilon is not None:
        calc.add("cpcm-solvation", cpcm_epsilon)
    if electronic_temperature is not None:
        calc.set("temperature", electronic_temperature * 3.166808578545117e-6) # K to Hartree
    if not verbosity:
        calc.set("verbosity", 0)
    calc.set("save-integrals", 1)
    res = calc.singlepoint()

    S = res.get("overlap-matrix")  # (nao, nao)
    H = res.get("hamiltonian-matrix")  # (nao, nao)
    P = res.get("density-matrix")  # (nao, nao) or (2, nao, nao)
    E = res.get("orbital-energies")  # (nao) or (2, nao)
    C = res.get("orbital-coefficients")  # (nao, nao) or (2, nao, nao)
    if spin_pol:
        F_a = S.dot(C[0]).dot(np.diag(E[0])).dot(LA.inv(C[0]))
        F_b = S.dot(C[1]).dot(np.diag(E[1])).dot(LA.inv(C[1]))
        res_dict = {"F_a": F_a, "F_b": F_b, "P_a": P[0], "P_b": P[1], "S": S, "H": H}
    else:
        F = S.dot(C).dot(np.diag(E)).dot(LA.inv(C))
        res_dict = {"F": F, "P": P, "S": S, "H": H}

    if get_energy:
        res_dict["energy"] = res.get("energy") # Ha
    if get_forces:
        res_dict["force"] = res.get("gradient") # Ha/bohr
    if get_dipole:
        res_dict["dipole"] = res.get("dipole") # elementary_charge*bohr
    if get_partial_charges:
        res_dict["partial_charges"] = res.get("charges") # e

    # Correcting orbital order for qcore's convention
    perm_map = _get_perm_map(element_numbers)

    if spin_pol:
        mat_keys = ["F_a", "F_b", "P_a", "P_b", "S", "H"]
    else:
        mat_keys = ["F", "P", "S", "H"]

    for m in mat_keys:
        if cutoff is not None:
            res_dict[m] = np.where(np.abs(res_dict[m]) <= cutoff, 0, res_dict[m])
        res_dict[m] = apply_perm_map(res_dict[m], perm_map)

    return res_dict

