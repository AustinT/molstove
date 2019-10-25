import shutil
import tempfile
from unittest import TestCase

from molstove import orca
from molstove.tools import Atom


class TestOrca(TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

        self.atoms = [
            Atom('H', 0, 0, 0),
            Atom('H', 0, 0, 1.2),
        ]

        self.charge = 0
        self.spin_multiplicity = 1

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_h2_single_point(self):
        c = orca.SinglePointCalculator(
            self.atoms,
            charge=self.charge,
            spin_multiplicity=self.spin_multiplicity,
            method='PBE',
            basis='def2-SVP',
            num_processes=1,
            base_dir=self.test_dir,
        )
        c.run()
        results = c.parse_results()
        self.assertAlmostEqual(results.energy, -1.10605458839)
        self.assertAlmostEqual(results.homo, -0.317061)
        self.assertAlmostEqual(results.lumo, -0.057809)

    def test_h2_opt(self):
        c = orca.StructureOptCalculator(
            self.atoms,
            charge=self.charge,
            spin_multiplicity=self.spin_multiplicity,
            method='PBE',
            basis='def2-SVP',
            num_processes=1,
            base_dir=self.test_dir,
        )
        c.run()
        results = c.parse_results()
        self.assertEqual(len(results.atoms), 2)
