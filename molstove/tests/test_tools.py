from unittest import TestCase

from molstove import tools, conformers


class TestTools(TestCase):
    def test_mol_to_smiles(self):
        mol = tools.mol_from_smiles('C')
        self.assertEqual(mol.GetNumAtoms(), 5)

    def test_failed_mol(self):
        with self.assertRaises(RuntimeError):
            tools.mol_from_smiles('XYZ')

    def test_conformer_to_atoms(self):
        mol = tools.mol_from_smiles('C')
        conformers.generate_conformers(mol, max_num_conformers=1)
        atoms = tools.conformer_to_atoms(mol, mol.GetConformer(0))

        self.assertEqual(len(atoms), 5)
        for atom in atoms:
            self.assertEqual(len(atom), 2)
            self.assertEqual(len(atom[1]), 3)
