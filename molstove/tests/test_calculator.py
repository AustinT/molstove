from unittest import TestCase

from molstove import calculator, tools, conformers


class TestSCF(TestCase):
    def test_h2(self):
        atoms = [
            ['H', (0, 0, 0)],
            ['H', (0, 0, 1.2)],
        ]

        c = calculator.Calculator(atoms=atoms, charge=0, basis='def2-SVP', xc='PBE')
        c.run()

        self.assertAlmostEqual(c.get_energy(), -1.1060139202294401)
        homo, lumo = c.get_homo_lumo()
        self.assertAlmostEqual(homo, -0.3176734201839017)
        self.assertAlmostEqual(lumo, -0.0570349631410537)
        self.assertAlmostEqual(c.get_dipole_moment(), 0)


class TestOptimizer(TestCase):
    def test_nh3(self):
        mol = tools.mol_from_smiles('N')
        conformers.generate_conformers(mol, max_num_conformers=5, seed=42)

        atoms_list = [tools.conformer_to_atoms(mol=mol, conformer=conformer) for conformer in mol.GetConformers()]

        for atoms in atoms_list:
            c1 = calculator.Calculator(atoms=atoms, charge=0, basis='def2-SVP', xc='PBE')
            c1.run()
            energy = c1.get_energy()

            atoms_opt = calculator.optimize(c1)

            c2 = calculator.Calculator(atoms=atoms_opt, charge=0, basis='def2-SVP', xc='PBE')
            c2.run()
            energy_opt = c2.get_energy()

            self.assertTrue(energy_opt <= energy)
