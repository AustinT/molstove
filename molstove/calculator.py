from typing import Tuple, List, Dict, Any

import numpy as np
from pyscf import gto, dft

from molstove import tools


class Calculator:
    def __init__(self, atoms, charge: int, basis: str, xc: str, unit='Ang'):
        """
        Construct PySCF calculator

        :param atoms: atoms (in unit <unit> and PySCF format)
        :param charge: charge of system
        :param basis: orbital basis
        :param xc: exchange correlation functional
        :param unit: unit of atoms (Ang or Bohr)
        """
        self.mol = gto.Mole()
        self.mol.atom = atoms
        self.mol.charge = charge
        self.mol.spin = 0  # singlet
        self.mol.basis = basis
        self.mol.build(unit=unit)

        # Density fitting not possible as nuclear gradients are not implemented
        self.mf = dft.RKS(self.mol)
        self.mf.verbose = 2
        self.mf.xc = xc

    def run(self) -> None:
        """
        Run SCF procedure
        """
        self.mf = self.mf.run()

        if not self.mf.converged:
            raise RuntimeError('SCF did not converge')

    def get_energy(self) -> float:
        """
        Get total electronic energy.

        :return: energy in Hartree
        """
        return self.mf.e_tot

    def get_homo_lumo(self) -> Tuple[float, float]:
        """
        Get HOMO and LUMO energies.

        :return: (e_homo, e_lumo) energies in Hartree
        """
        mo_energies = self.mf.mo_energy
        e_sorted = np.sort(mo_energies)
        num_occupied = self.mf.mol.nelectron // 2
        return e_sorted[num_occupied - 1], e_sorted[num_occupied]

    def get_dipole_moment(self) -> float:
        """
        Get norm of dipole moment vector.

        :return: dipole moment in debye
        """
        dipole_moment = self.mf.dip_moment(verbose=False)
        return np.sqrt(np.sum(np.square(dipole_moment)))


def optimize(calculator: Calculator) -> List[List]:
    """
    Perform structure optimization of molecule in calculator.

    :param calculator: built calculator
    :return: atomic positions (in Angstrom)
    """
    from pyscf.geomopt import optimize as berny_optimize
    atoms = berny_optimize(calculator.mf).atom  # in Bohr

    converted = []
    for element, coord in atoms:
        atom = [element, tuple(i * tools.ANGSTROM_PER_BOHR for i in coord)]
        converted.append(atom)

    return converted


def generate_report(c: Calculator) -> Dict[str, Any]:
    homo, lumo = c.get_homo_lumo()
    return {
        'atoms': c.mol.atom,
        'charge': c.mol.charge,
        'spin_multiplicity': c.mol.spin,
        'basis': c.mol.basis,
        'xc_functional': c.mf.xc,
        'energy': c.get_energy(),
        'homo': homo,
        'lumo': lumo,
    }
