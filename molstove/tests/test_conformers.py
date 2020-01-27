from unittest import TestCase

from rdkit.Chem.rdForceFieldHelpers import MMFFGetMoleculeProperties, MMFFGetMoleculeForceField

from molstove import conformers, tools


class TestConformers(TestCase):
    def setUp(self) -> None:
        self.smiles = 'CC([O-])=O'
        self.mol = tools.mol_from_smiles(self.smiles)

    def test_hs_added(self):
        mol = tools.mol_from_smiles('C')
        self.assertEqual(mol.GetNumAtoms(), 5)

    def test_invalid_molecule(self):
        with self.assertRaises(RuntimeError):
            tools.mol_from_smiles('c')

    def test_conformer_generation(self):
        smiles = 'C'
        mol = tools.mol_from_smiles(smiles)
        count = conformers.generate_conformers(mol, max_num_conformers=5, seed=42)
        self.assertEqual(count, 2)

    def test_conformer_generation_charged(self):
        count = conformers.generate_conformers(self.mol, max_num_conformers=5, seed=42)
        self.assertEqual(count, 5)

    def test_minimizer(self):
        conformers.generate_conformers(self.mol, max_num_conformers=5, seed=42)

        # Get original energies
        energies = []
        props = MMFFGetMoleculeProperties(self.mol)
        for i in range(self.mol.GetNumConformers()):
            potential = MMFFGetMoleculeForceField(self.mol, props, confId=i)
            energy = potential.CalcEnergy()
            energies.append(energy)

        min_energies = conformers.minimize_conformers(self.mol)
        self.assertEqual(self.mol.GetNumConformers(), len(min_energies))

        for e1, e2 in zip(energies, min_energies):
            self.assertTrue(e1 >= e2)

    def test_clusters(self):
        mol = tools.mol_from_smiles('CCCCC=C')
        conformers.generate_conformers(mol, max_num_conformers=25, seed=42)
        energies = conformers.minimize_conformers(mol)
        clusters = conformers.collect_clusters(mol, energies=energies)

        # 3 or 4 clusters (depends on version)
        self.assertIn(len(clusters), {3,4})
