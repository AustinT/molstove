from unittest import TestCase

from molstove import pyscf, tools, conformers
from molstove.tools import Atom


class TestPySCFSinglePoint(TestCase):
    def test_h2(self):
        atoms = [
            Atom('H', 0, 0, 0),
            Atom('H', 0, 0, 1.2),
        ]

        c = pyscf.Calculator(atoms=atoms, charge=0, basis='def2-SVP', xc='PBE')
        c.run()

        self.assertAlmostEqual(c.get_energy(), -1.1060139202294401)
        homo, lumo = c.get_homo_lumo()
        self.assertAlmostEqual(homo, -0.3176734201839017)
        self.assertAlmostEqual(lumo, -0.0570349631410537)
        self.assertAlmostEqual(c.get_dipole_moment(), 0)


class TestPySCFOptimizer(TestCase):
    def test_nh3(self):
        mol = tools.mol_from_smiles('N')
        conformers.generate_conformers(mol, max_num_conformers=5, seed=42)

        atoms_list = [tools.conformer_to_atoms(mol=mol, conformer=conformer) for conformer in mol.GetConformers()]

        for atoms in atoms_list:
            c1 = pyscf.Calculator(atoms=atoms, charge=0, basis='def2-SVP', xc='PBE')
            c1.run()
            energy = c1.get_energy()

            atoms_opt = pyscf.optimize(c1)

            c2 = pyscf.Calculator(atoms=atoms_opt, charge=0, basis='def2-SVP', xc='PBE')
            c2.run()
            energy_opt = c2.get_energy()

            self.assertTrue(energy_opt <= energy)
